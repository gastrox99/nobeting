// Drive yedek aracısı. Kurulum: DRIVE_YEDEK.md
// Komut dosyası özellikleri (Proje ayarları): ANAHTAR (gizli parola), KLASOR (Drive klasör kimliği).
var SAKLANAN = 30; // her kullanıcı için son 30 yedek

function doPost(e) {
  try {
    var props = PropertiesService.getScriptProperties();
    var p = JSON.parse(e.postData.contents);
    if (p.key !== props.getProperty('ANAHTAR')) return cevap({ok: false, error: 'anahtar'});
    JSON.parse(p.data); // bozuk içerik yazılmasın
    var klasor = DriveApp.getFolderById(props.getProperty('KLASOR'));
    var eski = klasor.getFilesByName(p.name);
    while (eski.hasNext()) eski.next().setTrashed(true);
    klasor.createFile(p.name, p.data, MimeType.PLAIN_TEXT);
    var hepsi = [], it = klasor.getFiles();
    while (it.hasNext()) {
      var f = it.next();
      if (f.getName().indexOf(p.prefix) === 0 && !f.isTrashed()) hepsi.push(f);
    }
    hepsi.sort(function (a, b) { return b.getName() < a.getName() ? -1 : 1; });
    for (var i = SAKLANAN; i < hepsi.length; i++) hepsi[i].setTrashed(true);
    return cevap({ok: true});
  } catch (err) {
    return cevap({ok: false, error: String(err)});
  }
}

function cevap(o) {
  return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON);
}
