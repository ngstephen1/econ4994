"""Publish already validated report files and preserve earlier report drafts."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pypdf import PdfReader

DIR=Path(__file__).resolve().parent
ROOT=DIR.parents[2]
BUILD=DIR.parent/'.step8b_progress_build'
BASE='professor_model_progress_8_slides'
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    originals=json.loads((DIR/'_build/protected_before.json').read_text())
    changed=[name for name,h in originals.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=h]
    assert not changed, changed
    receipt=json.loads((BUILD/f'finalization_{BASE}_v5.validation.json').read_text())
    archive=BUILD/'review_drafts';archive.mkdir(exist_ok=True)
    needs_publish=(DIR/f'{BASE}_v5.pptx').exists()
    validated_source=DIR/f'{BASE}_v5.pptx' if needs_publish else archive/f'{BASE}_v5.pptx'
    assert sha(validated_source)==receipt['finalSha256']
    for rev in (['', '_v2', '_v3', '_v4'] if needs_publish else []):
        for ext in ['.pptx','.pdf']:
            p=DIR/(BASE+rev+ext)
            if p.exists():
                target=archive/p.name
                if target.exists():raise FileExistsError(target)
                shutil.move(p,target)
    for ext in (['.pptx','.pdf'] if needs_publish else []):
        source=DIR/(BASE+'_v5'+ext)
        target=DIR/(BASE+ext)
        shutil.copyfile(source,target)
        assert sha(source)==sha(target)
        shutil.move(source,archive/source.name)
    for prefix in ['page','final-page']:
        for i in range(1,9):
            p=DIR/f'previews/pdf/{prefix}-{i}.png'
            if p.exists():shutil.move(p,archive/f'{prefix}-{i}.png')
    pptx=DIR/(BASE+'.pptx');pdf=DIR/(BASE+'.pdf')
    with zipfile.ZipFile(pptx) as z:
        slide_names=[f'ppt/slides/slide{i}.xml' for i in range(1,9)]
        assert len([n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml')])==8
        native=[]
        for n in slide_names:
            s=ET.fromstring(z.read(n))
            native.append({'slide':len(native)+1,'editable_text_shapes':len(s.findall('.//p:sp',NS)),
                           'native_tables':len(s.findall('.//a:tbl',NS)),
                           'connectors':len(s.findall('.//p:cxnSp',NS))})
        chart_names=[n for n in z.namelist() if '/charts/' in n and n.rsplit('/',1)[-1].startswith('chart') and n.endswith('.xml')]
        notes_names=[n for n in z.namelist() if n.startswith('ppt/notesSlides/notesSlide') and n.endswith('.xml')]
        assert len(chart_names)==1 and len(notes_names)==8
        assert native[2]['native_tables']==native[4]['native_tables']==1
        for i in [1,4,6]:assert native[i]['connectors']>0
        assert any(n.startswith('ppt/embeddings/') and n.endswith('.xlsx') for n in z.namelist())
    assert len(PdfReader(pdf).pages)==8
    # Image-backed PDF contains the exact reviewed slide pixels, not a new layout.
    from PIL import Image
    import numpy as np
    pixel_matches=[]
    reader=PdfReader(pdf)
    for i,page in enumerate(reader.pages,1):
        assert len(page.images)==1
        original=np.asarray(Image.open(DIR/f'previews/slide-{i:02d}.png').convert('RGB'))
        embedded=np.asarray(page.images[0].image.convert('RGB'))
        assert np.array_equal(original,embedded),i
        pixel_matches.append(i)
    evidence=json.loads((DIR/'evidence_manifest.json').read_text())
    # Keep the public manifest concise. The separate protected snapshot retains
    # the full preservation inventory. Data/record hashes remain in provenance.
    original_source_hashes=evidence.pop('source_hashes')
    source_paths={
      'docs/PROJECT_STATUS.md','docs/professor_whiteboard_model.md','docs/professor_loan_economics.md',
      'docs/professor_bank_regression.md','docs/professor_noise_robustness_full.md',
      'data/synthetic/professor_model/baseline/applicants_bank_view.csv',
      'data/synthetic/professor_model/baseline/simulation_truth.csv',
      'data/synthetic/professor_model/baseline/training_signals.csv',
      'data/synthetic/professor_model/baseline/risk_measurement_noise_private.csv',
      'data/synthetic/professor_model/baseline/applicant_splits.csv',
      'configs/professor_model/baseline.yaml',
      'results/metrics/professor_model/noise_robustness_full/manifest.json',
      'results/metrics/professor_model/noise_robustness_full/latest_report.json',
      'results/metrics/professor_model/noise_robustness_full/signal_reference_EVALUATOR_ONLY.json'}
    source_paths|={p for p in original_source_hashes if p.startswith('results/tables/professor_model/noise_robustness_full/')}
    source_paths|={f'results/tables/professor_model/{n}.csv' for n in ['bank_vs_oracle_summary','bank_vs_oracle_group_audit',
                   'group_aware_vs_blind_portfolio','group_aware_vs_blind_group_audit','group_aware_vs_blind_risk']}
    evidence['source_hashes']={p:original_source_hashes[p] for p in sorted(source_paths)}
    evidence['replication_hashes_manifest']='results/metrics/professor_model/noise_robustness_full/latest_report.json: replication_result_hashes'
    evidence['native_chart_rounding']='Slide4 workbook data rounded to4 decimals pp for Excel portability; labels2 decimals. Full values remain in verified_evidence.json.'
    evidence['final_output_hashes']={p.name:sha(p) for p in [pptx,pdf,DIR/'contact_sheet.png']}
    evidence['preservation_check']={'files_checked':len(originals),'changed':[],'scope':'all scientific source/config/test/docs and professor-model datasets/results'}
    (DIR/'evidence_manifest.json').write_text(json.dumps(evidence,indent=2))
    validation={'pptx_slides':8,'pdf_pages':8,'slide_size':'16:9, 1280×720 authoring pixels',
                'native_objects':native,'native_chart_count':1,'embedded_chart_workbook':True,'speaker_note_pages':8,
                'all_slides_visually_reviewed':True,'PDF_embedded_pixels_match_preview_slides':pixel_matches,
                'outliers_visible':[8,92],'preserved_scientific_files':len(originals),'changed_scientific_files':changed,
                'no_experiments_or_fits_run':True,'fresh_test_suite_run':False,'no_commit_or_push':True,
                'PDF_export':'144-dpi reviewed slide renders, image-backed read-only companion',
                'output_hashes':evidence['final_output_hashes'],'private_finalization_receipt':str(BUILD/f'finalization_{BASE}_v5.validation.json')}
    (DIR/'delivery_validation.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
