"""Copy existing professor-model CSVs into a verified sharing ZIP."""
from pathlib import Path
import csv
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    folders=['data/synthetic/professor_model','data/processed/professor_model','results/tables/professor_model']
    paths=sorted({p for name in folders for p in (ROOT/name).rglob('*.csv') if p.is_file()})
    paths+=sorted({p for name in folders[:2] for p in (ROOT/name).rglob('*metadata.json') if p.is_file()})
    full=ROOT/'results/metrics/professor_model/noise_robustness_full'
    paths+=[full/name for name in ['manifest.json','latest_report.json','signal_reference_EVALUATOR_ONLY.json']]
    assert len(paths)==len(set(paths))
    entries=[]
    for p in paths:
        relative=str(p.relative_to(ROOT))
        entry={'path':relative,'bytes':p.stat().st_size,'sha256':sha(p)}
        if p.suffix=='.csv':
            with p.open(newline='') as f:
                reader=csv.reader(f)
                entry['columns']=next(reader)
                entry['rows']=sum(1 for _ in reader)
        entries.append(entry)
    manifest={'scope':'Existing professor-model CSV datasets and results through Step 8B',
              'csv_files':sum(p.suffix=='.csv' for p in paths),
              'metadata_json_files':sum(p.suffix=='.json' for p in paths),
              'uncompressed_source_bytes':sum(e['bytes'] for e in entries),
              'no_data_regeneration_or_model_fitting':True,'files':entries}
    out=OUT/'professor_model_csv_bundle.zip'
    if out.exists():raise FileExistsError(out)
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,e in zip(paths,entries):z.write(p,e['path'])
        z.write(OUT/'README.txt','README.txt')
        z.writestr('FILE_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for p,e in zip(paths,entries):
            assert hashlib.sha256(z.read(e['path'])).hexdigest()==e['sha256']==sha(p)
    (OUT/'FILE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'archive':str(out),'csv_files':manifest['csv_files'],
                      'metadata_json_files':manifest['metadata_json_files'],
                      'archive_MB':round(out.stat().st_size/1e6,2),
                      'uncompressed_MB':round(manifest['uncompressed_source_bytes']/1e6,2),
                      'archive_integrity':'passed','original_files':'unchanged'},indent=2))

if __name__=='__main__':main()
