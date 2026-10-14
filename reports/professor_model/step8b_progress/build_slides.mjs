/** Editable academic deck, sourced only from audit_and_charts.py outputs. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
import { Presentation, PresentationFile, FileBlob } from '@oai/artifact-tool';

const DIR=path.dirname(fileURLToPath(import.meta.url));
const WORKSPACE=path.dirname(DIR);
const BUILD=path.join(WORKSPACE,'.step8b_progress_build');
const STEM=process.env.PROGRESS_DECK_STEM ?? 'professor_model_progress_8_slides';
const SKILL='/Users/macbook/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.11113/skills/presentations';
const PY='/Users/macbook/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const {finalizePresentation,resolvePresentationFont,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')));
const e=JSON.parse(await fs.readFile(path.join(DIR,'verified_evidence.json'),'utf8'));
const evidence=JSON.parse(await fs.readFile(path.join(DIR,'evidence_manifest.json'),'utf8'));
const font=resolvePresentationFont();
const C={maroon:'#861F41',orange:'#E5751F',slate:'#253746',gray:'#68737D',line:'#D7DBDF',light:'#F6F3F4',white:'#FFFFFF'};
const p=Presentation.create({slideSize:{width:1280,height:720}});
const num=(n,d=3)=>Number(n).toFixed(d);
let shapeIndex=0;
function text(s,t,x,y,w,h,size=26,color=C.slate,bold=false,align='left'){
 const sh=s.shapes.add({geometry:'textbox',name:`text-${++shapeIndex}`,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=t;sh.text.style={typeface:font,fontSize:size,color,bold,alignment:align,verticalAlignment:'top',insets:0,autoFit:'none',wrap:'square'};return sh;
}
function line(s,x,y,w,color=C.line,width=1){return s.shapes.add({geometry:'line',position:{left:x,top:y,width:w,height:0},fill:'none',line:{fill:color,width}});}
function slide(title,source){
 const s=p.slides.add();s.background.fill=C.white;
 text(s,title,56,39,1170,64,42,C.maroon,true);
 line(s,56,111,1168,C.line,1);
 text(s,source,56,684,1128,25,13.5,C.gray);
 text(s,String(p.slides.items.length).padStart(2,'0'),1208,684,32,24,14,C.gray,false,'right');return s;
}
function box(s,t,x,y,w,h,color=C.slate,size=24){
 const sh=s.shapes.add({geometry:'rect',name:`diagram-${++shapeIndex}`,position:{left:x,top:y,width:w,height:h},fill:C.light,line:{fill:color,width:1.3}});
 sh.text=t;sh.text.style={typeface:font,fontSize:size,color,alignment:'center',verticalAlignment:'middle',insets:10,autoFit:'none'};return sh;
}
function arrow(s,a,b,from='right',to='left'){
 return s.shapes.connect(a,b,{kind:'elbow',fromSide:from,toSide:to,line:{fill:C.gray,width:1.5},tail:{type:'triangle',width:'med',length:'med'}});
}
async function image(s,name,x,y,w,h){
 const bytes=await fs.readFile(path.join(DIR,'assets',name+'.png'));
 s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',alt:name.replaceAll('_',' '),fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
function table(s,values,x,y,w,h,widths,size=21){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
 t.styleOptions={headerRow:true,bandedRows:false};
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({
   textStyle:{typeface:font,fontSize:size,color:C.slate},margins:{left:8,right:8,top:10,bottom:10},anchor:'center',
   fill:C.white,borders:{fill:C.line,width:.5}});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
   const cell=t.getCell(r,c);cell.text.style={typeface:font,fontSize:size,color:r===0?C.white:C.slate,bold:r===0,alignment:c===0?'left':'right',verticalAlignment:'middle'};
   if(r===0)cell.fill=C.maroon;
 }
 for(let r=0;r<values.length;r++)t.rows[r].height=h/values.length;
 t.borders.assign({fill:C.line,width:.5});return t;
}
function notes(s,n,body){
 const refs=evidence.slides.filter(x=>x.slide===n);
 s.speakerNotes.text=body+'\n\nEvidence sources and computations:\n'+refs.map(x=>`${x.source}\nColumns: ${x.column}\nComputation: ${x.computation}\nDisplay rounding: ${x.rounding}`).join('\n\n');
}

// 1. A restrained title page with the requested executive evidence.
{
 const s=p.slides.add();s.background.fill=C.white;
 text(s,'Lending Under Imperfect Information',56,74,1168,88,54,C.maroon,true);
 text(s,'How noisy risk assessments affect lending decisions,\nprofitability, and group outcomes',56,185,1100,82,31);
 text(s,'Nguyen Phan Nguyen\nVirginia Tech — ECON 4994',56,299,700,64,24,C.gray);
 const xs=[56,450,846];
 ['100,000','200','91.5%'].forEach((v,i)=>text(s,v,xs[i],400,340,70,58,i===2?C.orange:C.maroon,true));
 ['Synthetic applicants','Independent noise\nreplications','Draws with a smaller absolute\ngroup approval gap when\nthe bank excludes B'].forEach((v,i)=>text(s,v,xs[i],480,350,98,24));
 text(s,'Noisy bank estimates can create substantial and unstable group lending\ndisparities even when the true-risk model has no direct group effect.',56,594,1168,62,25);
 text(s,'Synthetic simulation, not real-world mortgage evidence. Source: Step 8B, noise_full_results.csv',56,686,1135,23,14,C.gray);
 text(s,'01',1208,686,32,23,14,C.gray,false,'right');
 notes(s,1,'Full research title: Lending Under Imperfect Information: Risk Estimation, Group Disparities, and Credit Allocation. One fixed synthetic population. 183/200 independent measurement-noise draws yield a smaller absolute group approval gap under the blind policy. This descriptive frequency does not prove normative fairness.');
}
// 2. Editable information/accounting pipeline. No V1/V2 intervention enters.
{
 const s=slide('From True Risk to Lending Decisions','Source: professor-model Steps 2–7B, saved specifications and evaluation artifacts.');
 const applicants=box(s,'Synthetic applicants\nY, D, B, C',56,162,290,96);
 const truth=box(s,'Hidden true risk\ntrue_PD = f(Y, D, ε)',430,162,290,96);
 const target=box(s,'Noisy measurement PN\nPN = 0.8 true_PD + η\nη ~ Normal(0, SD=0.1)',864,162,360,96,C.maroon,22);
 arrow(s,applicants,truth);arrow(s,truth,target);
 const aware=box(s,'Group-aware OLS\nX = [Y, D, C, B]',625,341,285,91,C.maroon);
 const blind=box(s,'Group-blind OLS\nX = [Y, D, C]',939,341,285,91,C.orange);
 arrow(s,target,aware,'bottom','top');arrow(s,target,blind,'bottom','top');
 text(s,'bank_PD_raw = β̂₀ + β̂′X\nThe bank does not observe true_PD or ε.',56,345,515,77,25);
 text(s,'Working contract: T=30, θ=0.10, r=5% per period\nL=30θY. Fixed capital A=$1.826B.',56,437,750,46,21,C.gray);
 const risk=box(s,'Estimated default risk\np̂ = clip(raw, 0, 1)',864,491,360,92);
 arrow(s,aware,risk,'bottom','top');arrow(s,blind,risk,'bottom','top');
 const profit=box(s,'Expected loan profit\nΣ paymentₜ (1−p̂)ᵗ − L',430,491,340,92,C.slate,23);
 const cutoff=box(s,'Capital-constrained\ncommon lending cutoff',56,491,285,92,C.slate,23);
 arrow(s,risk,profit,'left','right');arrow(s,profit,cutoff,'left','right');
 const outcomes=box(s,'Approval decisions, group outcomes, and evaluator-only true expected profit',56,617,1168,49,C.maroon,24);
 arrow(s,cutoff,outcomes,'bottom','top');
 notes(s,2,'The simulator uses hidden truth to construct PN. Banks only fit on observed Y,D,C and, for the aware model, B plus an intercept. PN is a fitting target, not a feature. Raw predictions remain separate from clipped probabilities. PD means conditional per-period default in the working contract, so survival through payment t is (1−PD)^t. Bank policies optimize perceived expected profit first, then the evaluator values saved decisions with hidden truth. No realized default labels or V1/V2 mechanisms enter this family.');
}
// 3. Actual first-five ID join, with truth explicitly restricted to the evaluator.
{
 const s=slide('How We Simulated the Lending Data','Source: baseline bank-view, simulation_truth.csv, training_signals.csv. Researcher-defined synthetic distributions.');
 text(s,'100,000 applicants',56,152,430,45,32,C.maroon,true);
 text(s,'80,000 train / 20,000 evaluation',56,205,440,38,24);
 text(s,'Y   Annual income\nD   Debt-to-income ratio\nB   Binary group (15.07% B=1)\nC   Noisy credit signal\nε   Hidden borrower factor',56,270,444,184,25);
 text(s,'true_PD =\nsigmoid(0.52 − 0.65 log Y)\n× sigmoid(4 + 0.06 D + ε)',56,470,440,115,23);
 text(s,`Mean true_PD ${num(e.mean_true_PD*100,4)}%\nNoise SD 10 pp ≈ ${Math.round(e.noise_signal_SD_ratio)}× scaled-signal SD`,56,595,470,63,21,C.maroon,true);
 text(s,'EVALUATOR-ONLY DATA PREVIEW',526,163,700,35,24,C.maroon,true);
 text(s,'First five saved IDs. PD in percent, PN in raw units.',526,204,700,34,21,C.gray);
 const rows=[['Applicant ID','B','Y ($)','D','C','PD (%)','PN raw'],...e.snippet.map(r=>[r.applicant_id,String(r.B),Math.round(r.Y).toLocaleString('en-US'),num(r.D),num(r.C),num(r.true_PD*100,4),num(r.PN,4)])];
 const t=table(s,rows,526,263,700,326,[154,38,99,75,86,116,132],20);
 t.getCell(4,6).text.style={typeface:font,fontSize:20,color:C.orange,bold:true};
 text(s,'Negative PN is a noisy measurement, not negative true risk.\nNo empirical U.S. mortgage sample or representative preview.',526,610,700,56,21,C.gray);
 notes(s,3,'Exact preview IDs: '+e.snippet.map(r=>r.applicant_id).join(', ')+'. Values come from a 1:1 join of the three saved CSVs, sorted lexicographically, without selecting on outcomes. PN is the original canonical Step 4 draw, not a Monte Carlo target. Primitive distributions: log(Y)~N(11,SD .7), D~N(.35,SD .16), B~Bernoulli(.15), ε~N(0,SD1), ω~N(0,SD .2), C=.4ε+ω. Untruncated D includes 1,489 negative observations. True PD has no structural B term. The realized canonical SD(η)/SD(.8true_PD) is '+e.noise_signal_SD_ratio+'. Current economics uses constant conditional per-period default.');
}
// 4. Native chart with source-linked values. Canonical draw remains separate.
{
 const s=slide('Initial Bank, Group-Blind, and Oracle Results','Source: group_aware_vs_blind_portfolio.csv and group_aware_vs_blind_group_audit.csv, Steps 7A–7B.');
 text(s,'Single synthetic noise realization, not a robustness result',56,130,1168,38,24,C.gray);
 text(s,'Approval gap (B=1 minus B=0, pp)',56,193,650,36,25,C.slate,true);
 const chart=s.charts.add('bar',{
   position:{left:48,top:245,width:665,height:314},categories:['Group-aware','Group-blind','Oracle'],
   // Presentation rounding only, to match labels and Excel's stored precision.
   // Full source values remain in verified_evidence.json and the manifest.
   series:[{name:'Approval gap (pp)',values:e.canonical.map(r=>Number(num(r.approval_gap_pp,4))),fill:C.maroon,
            points:[{idx:0,fill:C.maroon},{idx:1,fill:C.orange},{idx:2,fill:C.gray}],valuesFormatCode:'+0.00;-0.00;0.00'}],
   barOptions:{direction:'column',grouping:'clustered',gapWidth:120},hasLegend:false,
   xAxis:{textStyle:{fontSize:21,fill:C.slate},position:'bottom',line:{fill:C.line,width:1}},
   yAxis:{min:-16,max:4,majorUnit:4,numberFormatCode:'0',textStyle:{fontSize:19,fill:C.gray},majorGridlines:{fill:'#E7E9EC',width:.6}},
   dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'+0.00;-0.00;0.00',textStyle:{fontSize:22,fill:C.slate,bold:true}},
   chartFill:C.white,plotAreaFill:C.white});
 applyPresentationChartFont(chart,{fontFamily:font});
 text(s,'True expected portfolio profit',800,194,425,38,25,C.slate,true);
 e.canonical.forEach((r,i)=>{
  const color=i===0?C.maroon:i===1?C.orange:C.gray;
  text(s,['Group-aware bank','Group-blind bank','True-risk oracle'][i],800,257+i*94,425,32,23,color);
  text(s,`$${num(r.true_expected_portfolio_profit/1e9,4)}B`,800,292+i*94,425,51,39,color,true);
 });
 text(s,'Excluding B changed the original approval gap from −14.39 pp to +0.16 pp.',56,579,1168,46,28,C.maroon,true);
 text(s,'Different risk rankings create different feasible cutoff portfolios.\nThe oracle is neither a fairness rule nor an unrestricted optimality bound.',56,630,1168,50,21,C.gray);
 notes(s,4,'Canonical noise uses seed 4994 child [5], while Step 8B uses master 499408 children [0] through [199]. The canonical draw is not included as replication 201. Exact true profits: aware $'+e.canonical[0].true_expected_portfolio_profit+', blind $'+e.canonical[1].true_expected_portfolio_profit+', oracle $'+e.oracle_true_profit+'. All three use the same saved capital $1,826,087,220.7466326, 20,000 evaluation requests, T30, theta .10, r .05 per period. Approvals are 3,900, 3,802, and 3,406. Group gaps are derived from saved group rates. Profit means nominal true expected profit, not realized profit. Signed oracle comparisons need not be nonnegative for every possible score-dependent cutoff family.');
}
// 5. Native experimental diagram and predeclared first-three table.
{
 const s=slide('Testing Whether the Result Survives New Noise','Source: Step 8B manifest, immutable replication records, noise_full_results.csv. Indices 0–199, master seed 499408.');
 const stages=[
  ['Frozen applicants, true risk, split,\ncontracts, capital, and oracle',158,90],
  ['200 independent η draws\n200 raw noisy PN targets',286,74],
  ['Group-aware OLS / Group-blind OLS',397,61],
  ['Re-optimize each common cutoff',495,61],
  ['Compare group outcomes and true profit',593,58]
 ];
 let previous;
 for(const [t,y,h] of stages){const sh=box(s,t,56,y,585,h,C.slate,24);if(previous)arrow(s,previous,sh,'bottom','top');previous=sh;}
 const countX=[735,948,1110],countW=[200,145,114];
 ['200/200','400','0'].forEach((t,i)=>text(s,t,countX[i],173,countW[i],55,43,i===2?C.gray:C.maroon,true));
 ['Successful\nreplications','Policy\nobservations','Scientific\nfailures'].forEach((t,i)=>text(s,t,countX[i],242,countW[i],61,22));
 text(s,'Only η changed',735,327,485,41,30,C.orange,true);
 text(s,'First three predeclared indices (gaps in pp)',735,384,485,36,22,C.gray);
 table(s,[['Draw','Aware','Blind','Blind−aware'],...e.first_three.map(r=>[String(r.replication),num(r.aware_gap_pp),num(r.blind_gap_pp),num(r.gap_change_pp)])],735,433,489,194,[82,110,104,193],22);
 text(s,'5 pilot reuses + 195 new replications',735,646,489,29,21,C.gray);
 notes(s,5,'Frozen population100000, split80000/20000, contract and original capital. Exactly200 distinct successful indices0..199,400 unique policy cells, no missing pairs or duplicates. All record/source/configuration hashes match. First5 reproduce pilot science exactly and are adopted without refitting. The canonical single draw remains a separate reference. Main run641.545seconds with tests alongside part of the run. Source excerpt uses indices0,1,2 only and takes paired values once per replication, not400 repeated paired values. Raw noise rows are reconstructible, not persisted as200 complete applicant datasets.');
}
// 6. Recreated distributions, readable at separate scales.
{
 const s=slide('Noisy Group-Aware Estimates Produce Unstable Gaps','Source: noise_full_results.csv, 200 draws. Brackets show p2.5–p97.5 empirical noise spreads, not causal intervals.');
 text(s,'The original gap changed direction across independent noise draws.',56,118,1168,30,22,C.gray);
 await image(s,'approval_gap_panels',42,143,670,420);
 await image(s,'coefficient_distribution',756,151,468,410);
 text(s,'Aware mean +8.840 pp, SD 36.206 [−46.658, +82.129]\nBlind mean +0.366 pp, SD 0.599 [−1.200, +1.454]',56,568,670,60,22);
 text(s,'52.5% positive / 47.5% negative B slopes\nNearly half of aware gaps were negative.',756,568,468,60,21);
 text(s,'91.5% had smaller blind absolute gaps',56,639,665,36,25,C.orange,true);
 text(s,'65.5% reached the original |14.39 pp| gap',756,639,475,36,23,C.maroon,true);
 notes(s,6,'Charts contain every replication. Gap panels deliberately use different x ranges to make the blind distribution legible. Aware p2.5/p97.5 −46.657711/+82.128929pp, MCSE2.560158pp. Blind p2.5/p97.5 −1.200024/+1.453878pp, MCSE.042370pp. Aware mean is asymmetric and median is0, with95/200negative,92positive,13zero reject-all cases. B mean.0000632779704111, SD.0010631270967191, MCSE.0000751744379353. Original B coefficient.0007105261442719142. The chart marks zero, canonical coefficient, and mean. Evaluator-only noise-free B projection '+e.signal_beta_B+' is small but not exactly zero. All200 OLS decompositions pass, maximum residual2.25514e−17. These finite-sample projections and descriptive noise frequencies neither prove intentional discrimination nor normative fairness.');
}
// 7. All outliers remain visible, with a clearly labeled central window.
{
 const s=slide('Profit Results Are Driven by Rare Extreme Cases','Source: noise_full_results.csv. Nominal true expected profits under stylized 30-period growing payments, not realized returns.');
 await image(s,'profit_full_range_and_zoom',42,151,710,414);
 text(s,'MEDIAN CHANGE',797,156,430,28,18,C.gray,true);
 text(s,'+$651,807',797,191,430,64,49,C.maroon,true);
 text(s,'Mean +$24.616M',797,266,430,38,27);
 text(s,'70.5% blind profit higher\n23.5% lower / 6.0% equal',797,315,430,69,24);
 text(s,'Two draws exceed +$2.36B each',797,399,430,33,24,C.orange,true);
 text(s,'Aware: zero loans in both\nBlind: 4,744 / 3,861 loans',797,440,430,67,23);
 text(s,'Reject-all: aware 13/200, blind 11/200\nZero block >90% capital: 17/200 each',797,516,430,58,21,C.gray);
 const ts=['Negative OLS\npredictions','Clip to zero','Indivisible\nzero-score block','Block exceeds\ncapital','Reject-all only\nfeasible cutoff'];
 let prev;
 ts.forEach((t,i)=>{const sh=box(s,t,56+i*239,591,212,61,C.slate,20);if(prev)arrow(s,prev,sh);prev=sh;});
 text(s,'The mean advantage does not represent the typical replication.',56,653,1168,28,22,C.maroon,true);
 notes(s,7,'Paired difference means blind minus aware true expected profit. Mean$24,616,229.7545, median$651,806.6808, SD$235,673,271.0845, MCSE$16,664,616.8128. Empirical p2.5/p97.5 −$4,279,050.0811/+7,122,265.8269. Extremes: replication8 +$2,361,202,289.1137 and92 +$2,365,919,379.4631. Full-range upper plot shows all200 points. Central lower window shows198/200 and explicitly identifies that the other2 occur above. No trimming/winsorization enters summaries. Aware13/200 and blind11/200 reject-all. Zero blocks would require >90% of capital in17/200 per policy, including infeasible blocks13aware/11blind. Those infeasible blocks consume no actual capital. Every nonempty prefix contains the lowest-risk zero block, so no nonempty cutoff is feasible when it is oversized. Clipping is not calibration. No risk ties split or budgets exceeded. Payment formula thetaY(1.05)^t, without costs or discounting, creates large stylized nominal profit levels.');
}
// 8. Final conclusions and separately authorized research recommendations.
{
 const s=slide('What We Learned and What Comes Next','Source: Step 8B audited records and current PROJECT_STATUS.md. Synthetic evidence only.');
 text(s,'Three main takeaways',56,160,526,42,29,C.maroon,true);
 const take=[['Noise matters','Very noisy measurements change fitted group\ncoefficients and lending outcomes.'],
 ['Group-blind usually reduced disparity','Smaller absolute gap in 91.5% of draws.\nEqual outcomes are not guaranteed.'],
 ['Economic interpretation needs care','Rare clipped-zero / reject-all cases\ndominate the mean profit difference.']];
 take.forEach(([a,b],i)=>{text(s,a,56,224+i*112,570,33,26,C.slate,true);text(s,b,56,266+i*112,570,63,24);});
 text(s,'Recommended next steps',705,160,521,42,29,C.maroon,true);
 const next=['Review noise SD 0.1 against the\nvery small underlying true-risk variation.',
 'Review clipping and the all-or-none\nzero-score cutoff behavior.',
 'Predeclare a matched B-coefficient\nintervention to separate refitting effects.',
 'Then predeclare alternative noise levels\nand independent new populations.'];
 next.forEach((t,i)=>{text(s,String(i+1).padStart(2,'0'),705,224+i*87,43,36,24,C.orange,true);text(s,t,766,224+i*87,460,64,24);});
 text(s,'Next goal: isolate mechanisms before broader claims\nabout discrimination or ML improvements.',56,575,1168,64,29,C.maroon,true);
 text(s,'No intentional discrimination established. Step 8B complete.\nLatest documented suite: 624 passed, 4 unrelated V2 failures, 32 warnings.',56,643,1168,39,17,C.gray);
 notes(s,8,'Future steps are recommendations only and require separate scientific specifications. The current study fixes one population and varies onlyη. RemovingB changes all re-estimated slopes, bounded ties, and optimal cutoffs, so this comparison does not isolate a pureB-coefficient effect. An intervention could hold other coefficients fixed. No flexible ML comparisons, empirical HMDA analysis, systemic/direct interventions, or new-population study are part of the completed professor family. The latest documented full suite is624passed4preexistingV2checkpointfilename failures32warnings, not a fresh test execution for this report. Professor suite344passedwithoutwarnings. Historical Step8A heading and Step1annual-rate suggestion are retained historical evidence; current validated full records and implemented per-period accounting govern this deck.');
}

if(p.slides.items.length!==8)throw new Error('Exactly eight slides required');
await fs.mkdir(BUILD,{recursive:true});
await fs.writeFile(path.join(BUILD,'authoring-inspection.ndjson'),(await p.inspect({kind:'slide,textbox,chart,table,notes',maxChars:200000})).ndjson);
const candidate=path.join(BUILD,`candidate_${STEM}.pptx`);
await (await PresentationFile.exportPptx(p)).save(candidate);
const finalPath=path.join(DIR,STEM+'.pptx');
const receipt=await finalizePresentation({workspaceDir:WORKSPACE,candidatePath:candidate,finalPath,
 pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 explicitTotalSlideCount:8,requiredNativeTableOwnerSlides:[3,5],requiredNativeChartOwnerSlides:[4],
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','3','--require-native-table-slide','5'],
 receiptPath:path.join(BUILD,`finalization_${STEM}.validation.json`)});
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<8;i++){
 const slide=final.slides.items[i];
 const blob=await final.export({slide,format:'png',scale:1.5});
 await fs.writeFile(path.join(DIR,`previews/slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const montage=await final.render({montage:{columns:2,slideWidth:640,padding:16,gap:16,format:'png'}});
await fs.writeFile(path.join(DIR,'contact_sheet.png'),new Uint8Array(await montage.arrayBuffer()));
const pdfPath=path.join(DIR,STEM+'.pdf');
// Direct artifact PDF export lacks its #standard-fonts/Helvetica module in
// this runtime. Bundled LibreOffice changes fonts and chart label positions.
// Use the verified final-PPTX renders for exact visual agreement instead.
const pdfRenderer='verified_final_PPTX_slide_renders_144dpi';
const conversion=spawnSync(PY,[path.join(DIR,'export_matching_pdf.py'),'--output',pdfPath],{encoding:'utf8',timeout:60000});
if(conversion.status!==0)throw new Error('Matching PDF export failed: '+conversion.stderr);
await fs.writeFile(path.join(DIR,'rendering_metadata.json'),JSON.stringify({font,slideCount:8,pdfRenderer,
 nativeTablesOnSlides:[3,5],nativeChartsOnSlides:[4],editableDiagramSlides:[2,5,7],
 PDF_note:'Image-backed read-only PDF, 144 dpi. Native editable objects remain in the PPTX. All signs and layouts match the final deck renders.',
 chartRounding:{slide4NativeValues:'4 decimals in percentage points; labels 2 decimals; unrounded originals retained in evidence JSON'}},null,2));
console.log(JSON.stringify({font,slides:8,finalPath,pdf:pdfPath,pdfRenderer,validationReceipt:path.join(BUILD,`finalization_${STEM}.validation.json`)},null,2));
