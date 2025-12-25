# -*- coding: utf-8 -*-
"""
OR-Tools CP-SAT Solver for Fair Shift Scheduling
Uses constraint programming for optimal shift distribution
"""

from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SolverResult:
    success: bool
    schedule: Optional[pd.DataFrame]
    message: str
    solve_time: float
    stats: Dict


def solve_schedule(
    isimler: List[str],
    sutunlar: List[str],
    gun_detaylari: Dict,
    kisi_sayisi: int,
    min_bosluk: int,
    df_unwanted: pd.DataFrame,
    df_preferred: pd.DataFrame,
    forbidden_pairs: List[Tuple[str, str]],
    person_limits: Dict[str, Tuple[int, int]],
    rol_isimleri: List[str],
    timeout_seconds: int = 30
) -> SolverResult:
    """
    Solve shift scheduling using OR-Tools CP-SAT solver.
    
    Args:
        isimler: List of team member names
        sutunlar: List of day columns (e.g., ['1_Pzt', '2_Sal', ...])
        gun_detaylari: Dict with day info (weekend, holiday, day_num)
        kisi_sayisi: Number of people per shift
        min_bosluk: Minimum days between shifts for same person
        df_unwanted: DataFrame of unavailable days (True = can't work)
        df_preferred: DataFrame of preference scores (higher = more preferred)
        forbidden_pairs: List of (person1, person2) who can't work together
        person_limits: Dict of person -> (min_shifts, max_shifts)
        rol_isimleri: List of role names
        timeout_seconds: Max solving time
    
    Returns:
        SolverResult with schedule DataFrame and statistics
    """
    import time
    start_time = time.time()
    
    num_people = len(isimler)
    num_days = len(sutunlar)
    num_roles = kisi_sayisi
    
    if num_people == 0 or num_days == 0:
        return SolverResult(
            success=False,
            schedule=None,
            message="Ekip veya gün sayısı sıfır olamaz",
            solve_time=0,
            stats={}
        )
    
    model = cp_model.CpModel()
    
    shifts = {}
    for p_idx, person in enumerate(isimler):
        for d_idx, day in enumerate(sutunlar):
            for r in range(num_roles):
                shifts[(p_idx, d_idx, r)] = model.NewBoolVar(f'shift_p{p_idx}_d{d_idx}_r{r}')
    
    for d_idx, day in enumerate(sutunlar):
        for r in range(num_roles):
            model.Add(sum(shifts[(p_idx, d_idx, r)] for p_idx in range(num_people)) == 1)
    
    for p_idx, person in enumerate(isimler):
        for d_idx, day in enumerate(sutunlar):
            model.Add(sum(shifts[(p_idx, d_idx, r)] for r in range(num_roles)) <= 1)
    
    for p_idx, person in enumerate(isimler):
        for d_idx, day in enumerate(sutunlar):
            if person in df_unwanted.index and day in df_unwanted.columns:
                if df_unwanted.at[person, day]:
                    for r in range(num_roles):
                        model.Add(shifts[(p_idx, d_idx, r)] == 0)
    
    if min_bosluk > 0:
        for p_idx in range(num_people):
            for d_idx in range(num_days - min_bosluk):
                window_vars = []
                for d in range(d_idx, min(d_idx + min_bosluk + 1, num_days)):
                    for r in range(num_roles):
                        window_vars.append(shifts[(p_idx, d, r)])
                model.Add(sum(window_vars) <= 1)
    
    person_to_idx = {name: idx for idx, name in enumerate(isimler)}
    for (p1, p2) in forbidden_pairs:
        if p1 in person_to_idx and p2 in person_to_idx:
            p1_idx = person_to_idx[p1]
            p2_idx = person_to_idx[p2]
            for d_idx in range(num_days):
                p1_works = model.NewBoolVar(f'p1works_{p1_idx}_{d_idx}')
                p2_works = model.NewBoolVar(f'p2works_{p2_idx}_{d_idx}')
                model.Add(sum(shifts[(p1_idx, d_idx, r)] for r in range(num_roles)) == 1).OnlyEnforceIf(p1_works)
                model.Add(sum(shifts[(p1_idx, d_idx, r)] for r in range(num_roles)) == 0).OnlyEnforceIf(p1_works.Not())
                model.Add(sum(shifts[(p2_idx, d_idx, r)] for r in range(num_roles)) == 1).OnlyEnforceIf(p2_works)
                model.Add(sum(shifts[(p2_idx, d_idx, r)] for r in range(num_roles)) == 0).OnlyEnforceIf(p2_works.Not())
                model.AddBoolOr([p1_works.Not(), p2_works.Not()])
    
    for person, (min_s, max_s) in person_limits.items():
        if person in person_to_idx:
            p_idx = person_to_idx[person]
            total_shifts = sum(shifts[(p_idx, d_idx, r)] 
                             for d_idx in range(num_days) 
                             for r in range(num_roles))
            model.Add(total_shifts >= min_s)
            model.Add(total_shifts <= max_s)
    
    if not person_limits:
        expected_per_person = (num_days * num_roles) // num_people
        min_shifts_global = max(0, expected_per_person - 2)
        max_shifts_global = expected_per_person + 3
        
        for p_idx in range(num_people):
            total_shifts = sum(shifts[(p_idx, d_idx, r)] 
                             for d_idx in range(num_days) 
                             for r in range(num_roles))
            model.Add(total_shifts >= min_shifts_global)
            model.Add(total_shifts <= max_shifts_global)
    
    weekend_days = [d_idx for d_idx, day in enumerate(sutunlar) if gun_detaylari[day]['weekend']]
    for p_idx in range(num_people):
        for i in range(len(weekend_days) - 1):
            d1, d2 = weekend_days[i], weekend_days[i + 1]
            if d2 - d1 <= 2:
                works_d1 = sum(shifts[(p_idx, d1, r)] for r in range(num_roles))
                works_d2 = sum(shifts[(p_idx, d2, r)] for r in range(num_roles))
                model.Add(works_d1 + works_d2 <= 1)
    
    preference_terms = []
    for p_idx, person in enumerate(isimler):
        for d_idx, day in enumerate(sutunlar):
            pref_val = df_preferred.at[person, day] if person in df_preferred.index else 0
            for r in range(num_roles):
                preference_terms.append(shifts[(p_idx, d_idx, r)] * pref_val)
    
    shift_counts = []
    for p_idx in range(num_people):
        count = model.NewIntVar(0, num_days * num_roles, f'count_{p_idx}')
        model.Add(count == sum(shifts[(p_idx, d_idx, r)] 
                              for d_idx in range(num_days) 
                              for r in range(num_roles)))
        shift_counts.append(count)
    
    max_count = model.NewIntVar(0, num_days * num_roles, 'max_count')
    min_count = model.NewIntVar(0, num_days * num_roles, 'min_count')
    model.AddMaxEquality(max_count, shift_counts)
    model.AddMinEquality(min_count, shift_counts)
    
    spread = model.NewIntVar(0, num_days * num_roles, 'spread')
    model.Add(spread == max_count - min_count)
    
    model.Minimize(spread * 1000 - sum(preference_terms))
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_seconds
    solver.parameters.num_search_workers = 4
    
    status = solver.Solve(model)
    
    solve_time = time.time() - start_time
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        schedule_data = {}
        for d_idx, day in enumerate(sutunlar):
            assigned = []
            for r in range(num_roles):
                for p_idx, person in enumerate(isimler):
                    if solver.Value(shifts[(p_idx, d_idx, r)]) == 1:
                        assigned.append(person)
                        break
            schedule_data[day] = assigned
        
        schedule_df = pd.DataFrame(schedule_data)
        if num_roles > 0:
            schedule_df.index = rol_isimleri[:num_roles] if rol_isimleri else [f'Rol{i+1}' for i in range(num_roles)]
        
        counts = {person: 0 for person in isimler}
        for p_idx, person in enumerate(isimler):
            for d_idx in range(num_days):
                for r in range(num_roles):
                    if solver.Value(shifts[(p_idx, d_idx, r)]) == 1:
                        counts[person] += 1
        
        count_values = list(counts.values())
        std_dev = np.std(count_values) if count_values else 0
        
        status_text = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
        
        return SolverResult(
            success=True,
            schedule=schedule_df,
            message=f"✅ {status_text} çözüm bulundu! (Süre: {solve_time:.2f}s)",
            solve_time=solve_time,
            stats={
                'status': status_text,
                'objective': solver.ObjectiveValue(),
                'spread': solver.Value(spread),
                'shift_counts': counts,
                'std_dev': std_dev,
                'conflicts': solver.NumConflicts(),
                'branches': solver.NumBranches()
            }
        )
    else:
        status_names = {
            cp_model.INFEASIBLE: "INFEASIBLE (Kısıtlar karşılanamıyor)",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.UNKNOWN: "UNKNOWN (Zaman aşımı veya belirsiz)"
        }
        return SolverResult(
            success=False,
            schedule=None,
            message=f"❌ Çözüm bulunamadı: {status_names.get(status, 'UNKNOWN')}",
            solve_time=solve_time,
            stats={'status': status_names.get(status, 'UNKNOWN')}
        )
