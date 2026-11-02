/**
 * Build the 14-slide professor follow-up from already validated saved results.
 * This script never generates applicants, fits models, or selects loans.
 *
 * Run with the bundled Node runtime. Override RUNTIME_NODE_MODULES,
 * PRESENTATIONS_SKILL_DIR, RUNTIME_PYTHON, or REFERENCE_PDF when needed.
 * Each NOV2_SLIDE_REVISION writes a separate immutable candidate/delivery.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const WORKSPACE = path.join(ROOT, 'reports/professor_model/nov2_followup');
const BUILD = path.join(WORKSPACE, '_build');
const DELIVERY = path.join(WORKSPACE, 'delivery');
const REV = process.env.NOV2_SLIDE_REVISION ?? '01';
if (!/^\d{2}$/.test(REV)) throw new Error('NOV2_SLIDE_REVISION must contain two digits');
const STEM = `professor_nov2_followup_${REV}`;
const REFERENCE = process.env.REFERENCE_PDF ?? '/Users/macbook/Desktop/nov_2_followup_slide_update.pptx.pdf';
const MODULES = process.env.RUNTIME_NODE_MODULES ?? '/Users/macbook/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const SKILL = process.env.PRESENTATIONS_SKILL_DIR ?? '/Users/macbook/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const PY = process.env.RUNTIME_PYTHON ?? '/Users/macbook/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const {Presentation, PresentationFile, FileBlob} = await import(pathToFileURL(path.join(MODULES, '@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {finalizePresentation, resolvePresentationFont, applyPresentationChartFont} = await import(pathToFileURL(path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')).href);
const TABLES = path.join(ROOT, 'results/tables/professor_model/nov2_followup');
const METRICS = path.join(ROOT, 'results/metrics/professor_model/nov2_followup');
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

function command(executable, args) {
  const result = spawnSync(executable, args, {encoding:'utf8', maxBuffer:30e6});
  if (result.status !== 0) throw new Error(`${executable} failed: ${result.stderr || result.stdout}`);
  return result.stdout;
}
function readCSV(filename) {
  return JSON.parse(command(PY, ['-c', 'import csv,json,sys; print(json.dumps(list(csv.DictReader(open(sys.argv[1],newline="")))))', filename]));
}
const requiredFiles = ['group_results.csv', 'portfolio_results.csv', 'risk_diagnostics.csv', 'penalty_comparisons.csv', 'ml_validation.csv', 'systemic_risk_audit.csv'];
const inputs = {};
for (const name of requiredFiles) inputs[path.relative(ROOT,path.join(TABLES,name))] = sha(await fs.readFile(path.join(TABLES,name)));
for (const name of ['run_manifest.json','validation.json','bank_freeze_checkpoint.json','bank_selections.json']) {
  inputs[path.relative(ROOT,path.join(METRICS,name))] = sha(await fs.readFile(path.join(METRICS,name)));
}
const groups = readCSV(path.join(TABLES,'group_results.csv'));
const portfolios = readCSV(path.join(TABLES,'portfolio_results.csv'));
const diagnostics = readCSV(path.join(TABLES,'risk_diagnostics.csv'));
const comparisons = readCSV(path.join(TABLES,'penalty_comparisons.csv'));
const systemicAudit = readCSV(path.join(TABLES,'systemic_risk_audit.csv'));
const validation = JSON.parse(await fs.readFile(path.join(METRICS,'validation.json'),'utf8'));
const runManifest = JSON.parse(await fs.readFile(path.join(METRICS,'run_manifest.json'),'utf8'));
const selections = JSON.parse(await fs.readFile(path.join(METRICS,'bank_selections.json'),'utf8'));
if(validation.status!=='validated' || runManifest.status!=='complete_two_scenario_single_realization' ||
   validation.all_six_exact_cutoffs_verified!==true || validation.bank_freeze_verified!==true) {
  throw new Error('Only a completed, validated six-policy scientific run may be presented');
}
for(const [relative,digest] of Object.entries(inputs)) {
  if(relative.endsWith('/run_manifest.json')) continue; // manifests do not hash themselves
  if(runManifest.artifacts[relative]!==digest) throw new Error(`Manifest input binding differs: ${relative}`);
}
const referenceHash = sha(await fs.readFile(REFERENCE));
const scenarios = ['scenario_1_no_systemic','scenario_2_systemic'];
const methods = ['ols_control','ols_penalty','ml_histgb'];
const labels = ['OLS\ncontrol','OLS\n+ penalty','Gradient\nboosting'];

// Column names here match the experiment's public CSV contract. Reject
// missing metrics instead of silently substituting a differently scaled field.
const col = {
  n:'n_applicants', count:'number_approved', rate:'approval_rate',
  groupProfit:'true_expected_portfolio_profit_USD', profit:'true_expected_portfolio_profit_USD',
};
function num(row,key) {
  if (!(key in row) || row[key] === '' || !Number.isFinite(Number(row[key]))) throw new Error(`Missing/invalid ${key}`);
  return Number(row[key]);
}
function group(s,m,b) {
  const found=groups.filter(r=>r.scenario_id===s&&r.method_id===m&&Number(r.B)===b);
  if(found.length!==1) throw new Error(`Expected one group cell: ${s}/${m}/${b}`);
  return found[0];
}
function portfolio(s,m) {
  const found=portfolios.filter(r=>r.scenario_id===s&&r.method_id===m);
  if(found.length!==1) throw new Error(`Expected one portfolio cell: ${s}/${m}`);
  return found[0];
}
if(groups.length!==12 || portfolios.length!==6) throw new Error('Expected twelve group and six portfolio rows');
for(const s of scenarios) for(const m of methods) {
  const a=group(s,m,0), b=group(s,m,1), p=portfolio(s,m);
  if(num(a,col.n)+num(b,col.n)!==20000) throw new Error('Evaluation counts must total 20,000');
  if(Math.abs(num(a,col.groupProfit)+num(b,col.groupProfit)-num(p,col.profit))>.00001) throw new Error('Group profits do not sum to portfolio value');
  for(const r of [a,b]) if(num(r,col.rate)<0 || num(r,col.rate)>1) throw new Error('Approval rate outside [0,1]');
}

await fs.mkdir(BUILD,{recursive:true});
await fs.mkdir(DELIVERY,{recursive:true});
const outPptx=path.join(DELIVERY,`${STEM}.pptx`);
try { await fs.access(outPptx); throw new Error('Refusing to overwrite existing finalized PPTX'); }
catch(error) { if(error.code!=='ENOENT') throw error; }
const referencePrefix=path.join(BUILD,'reference');
command('pdftoppm',['-r','144','-f','9','-l','14','-png',REFERENCE,referencePrefix]);

const font=resolvePresentationFont({sourceFont:'Helvetica Neue'});
const C={maroon:'#861F41',orange:'#E5751F',teal:'#237C83',slate:'#253746',gray:'#68737D',line:'#D7DBDF',light:'#F6F3F4',white:'#FFFFFF'};
const P=Presentation.create({slideSize:{width:1280,height:720}});
let serial=0;
const speaking=[];
const f=(v,d=2)=>Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const signed=(v,d=2)=>`${v<0?'−':v>0?'+':''}${f(Math.abs(v),d)}`;
const dollars=v=>`${v<0?'−':v>0?'+':''}$${f(Math.abs(v)/1e6,3)}M`;
const cv=v=>Number(Number(v).toFixed(6));
function text(s,value,x,y,w,h,size=26,color=C.slate,bold=false,align='left') {
  const sh=s.shapes.add({geometry:'textbox',name:`text-${++serial}`,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  sh.text=String(value);
  sh.text.style={typeface:font,fontSize:size,color,bold,alignment:align,verticalAlignment:'top',insets:0,autoFit:'none',wrap:'square'};
  return sh;
}
function line(s,x,y,w) { s.shapes.add({geometry:'line',position:{left:x,top:y,width:w,height:0},fill:'none',line:{fill:C.line,width:1}}); }
function slide(title,source) {
  const s=P.slides.add();s.background.fill=C.white;
  text(s,title,56,38,1168,62,42,C.maroon,true);line(s,56,112,1168);
  text(s,source,56,682,1134,27,13,C.gray);
  text(s,String(P.slides.items.length).padStart(2,'0'),1208,682,32,24,14,C.gray,false,'right');
  return s;
}
function note(s,body,sources=[]) {
  const refs=[...sources, 'Single matched synthetic realization. No empirical mortgage-discrimination claim.'];
  s.speakerNotes.text=body+'\n\nEvidence and interpretation\n'+refs.join('\n');
  speaking.push({slide:P.slides.items.length,body,sources:refs});
}
function box(s,body,x,y,w,h,color=C.slate,size=25) {
  s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:C.light,line:{fill:color,width:1.2}});
  text(s,body,x+12,y+12,w-24,h-24,size,color,false,'center');
}
function arrow(s,x,y,w,h,direction='right') {
  s.shapes.add({geometry:direction==='down'?'downArrow':direction==='left'?'leftArrow':'rightArrow',position:{left:x,top:y,width:w,height:h},fill:C.gray,line:{fill:'none',width:0}});
}
function table(s,values,x,y,w,h,widths,size=24) {
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
  t.styleOptions={headerRow:true,bandedRows:false};
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({textStyle:{typeface:font,fontSize:size,color:C.slate},margins:{left:12,right:12,top:12,bottom:12},anchor:'center',fill:C.white,borders:{fill:C.line,width:.6}});
  for(let r=0;r<values.length;r++) for(let c=0;c<values[0].length;c++) {
    const cell=t.getCell(r,c);
    cell.text.style={typeface:font,fontSize:size,color:r===0?C.white:C.slate,bold:r===0,alignment:'left',verticalAlignment:'middle'};
    if(r===0) cell.fill=C.maroon;
  }
  for(let r=0;r<values.length;r++) t.rows[r].height=h/values.length;
  t.borders.assign({fill:C.line,width:.6});return t;
}
const axis={textStyle:{fontSize:19,fill:C.gray},line:{fill:C.line,width:1}};
const allProfit=groups.map(r=>num(r,col.groupProfit)/1e9);
const profitMin=Math.min(0,Math.floor(Math.min(...allProfit)*2)/2);
const profitMax=Math.max(.5,Math.ceil(Math.max(...allProfit)*2)/2);
function resultChart(s,scenario,kind,x) {
  const approval=kind==='approval';
  const series=[0,1].map(b=>({name:`B=${b}`,fill:b===0?C.slate:C.orange,values:methods.map(m=>cv(approval?num(group(scenario,m,b),col.rate):num(group(scenario,m,b),col.groupProfit)/1e9))}));
  const chart=s.charts.add('bar',{
    position:{left:x,top:235,width:568,height:354},categories:labels,series,
    hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{fontSize:20,fill:C.slate}},
    barOptions:{direction:'column',grouping:'clustered',gapWidth:40,overlap:0},
    xAxis:{...axis,tickLabelPosition:'low'},
    yAxis:{...axis,min:approval?0:profitMin,max:approval?1:profitMax,majorUnit:approval?.2:.5,numberFormatCode:approval?'0%':'0.0',majorGridlines:{fill:C.line,width:.7}},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:approval?'0.0%':'0.000',textStyle:{fontSize:17.5,fill:C.slate,bold:true}},
    chartFill:C.white,plotAreaFill:C.white,
  });
  applyPresentationChartFont(chart,{fontFamily:font});
  return chart;
}
function resultSlide(scenario,number) {
  const s=slide(`Scenario ${number}: Approval Rates and True Profit`,'Source: nov2_followup/group_results.csv and portfolio_results.csv. Same contracts and capital.');
  const mlRejectAll=String(portfolio(scenario,'ml_histgb').reject_all_selected).toLowerCase()==='true';
  const mlUtilization=num(portfolio(scenario,'ml_histgb'),'capital_utilization');
  const mlLimited=selections[scenario].ml_histgb.selection.capital_limits_next_cutoff===true;
  const caption=mlRejectAll?'Gradient boosting selects reject-all under the common-cutoff rule':
    mlUtilization<.99&&mlLimited?`ML uses ${f(100*mlUtilization,1)}% of capital. The next whole score group exceeds budget`:
    '20,000 held-out applicants, one matched noise realization';
  text(s,caption,56,147,1168,37,25);
  text(s,'Within-group approval rate',56,195,568,40,27,C.slate,true);
  text(s,'True expected profit by group ($B)',656,195,568,40,27,C.slate,true);
  resultChart(s,scenario,'approval',56);resultChart(s,scenario,'profit',656);
  const control=portfolio(scenario,'ols_control'),penalty=portfolio(scenario,'ols_penalty');
  const gap=m=>100*(num(group(scenario,m,1),col.rate)-num(group(scenario,m,0),col.rate));
  text(s,`Penalty minus control: true portfolio profit ${dollars(num(penalty,col.profit)-num(control,col.profit))}`,56,610,1168,35,25,C.maroon,true);
  const b0=num(group(scenario,'ols_control',0),col.n),b1=num(group(scenario,'ols_control',1),col.n);
  text(s,`Profit bars sum funded-loan contributions. Group sizes: B=0 ${f(b0,0)} and B=1 ${f(b1,0)}.`,56,651,1168,25,17.5,C.gray);
  note(s,`Here are the actual results for Scenario ${number}. The approval chart uses all applicants in each group as its denominator. The profit chart sums true expected bank profit from funded loans in each group. The three methods use the same capital budget and contract. The penalty-versus-control true portfolio profit difference is ${dollars(num(penalty,col.profit)-num(control,col.profit))}. B=1 minus B=0 approval gaps are ${methods.map(m=>`${m}: ${signed(gap(m))} percentage points`).join(', ')}. The ML policy uses ${f(100*mlUtilization,4)} percent of capital. ${mlLimited?'Its next complete tied-score block exceeds the remaining budget, so the exact common cutoff cannot fund that whole block. ':''}${mlRejectAll?'The gradient-boosting cutoff selects reject-all; this is preserved and reported rather than changing the model or tie rule to force lending. ':''}These are observed outcomes, not imposed validation targets. Higher approval counts do not imply higher total capital use or profit. Group profit totals reflect the unequal group sizes, so the accompanying table also reports profit per eligible applicant.`,[
    'results/tables/professor_model/nov2_followup/group_results.csv; filter scenario_id='+scenario+'. Twelve rows across two scenarios, three methods and two groups.',
    'results/tables/professor_model/nov2_followup/portfolio_results.csv; group true profits sum to each method’s portfolio true profit.',
    'results/tables/professor_model/nov2_followup/penalty_comparisons.csv. Expected synthetic USD under the undiscounted stylized growing-payment contract, not realized annual returns.',
    'results/metrics/professor_model/nov2_followup/bank_selections.json: ML capital_limits_next_cutoff and exact tied-score policy. The same capital ceiling is available to every method; utilization can differ.',
    'The exact bounded-PD cutoff preserves indivisible score ties and reject-all. No hidden truth selects models, cutoffs or presentation examples.'
  ]);
}

// 1. Preserve the reference's title composition while identifying the new study.
{
  const s=P.slides.add();s.background.fill=C.white;
  text(s,'Lending Under Imperfect Information',56,74,1168,88,54,C.maroon,true);
  text(s,'November 2 follow-up: two scenarios and three methods',56,185,1168,66,32);
  text(s,'Nguyen Phan Nguyen\nVirginia Tech - ECON 4994\n2 November 2026',56,281,1000,108,24,C.gray);
  text(s,'Corrected +0.01B penalty\nSystemic risk and gradient-boosting comparison',56,447,1168,114,35,C.maroon,true);
  text(s,'100,000 applicants. 80,000 training. 20,000 evaluation.',56,598,1168,42,25);
  text(s,'Single matched synthetic run. Earlier studies appear only in the historical appendix.',56,682,1134,27,14,C.gray);
  text(s,'01',1208,682,32,24,14,C.gray,false,'right');
  note(s,'This update implements the changes from our November 2 discussion. The manual adjustment is now plus 0.01 times B, which increases perceived default risk for B=1. I compare an unadjusted OLS model, the same OLS score with the penalty, and gradient boosting. I evaluate both the original no-systemic world and a world with a one-standard-deviation upward epsilon shift for B=1. The original population, noise and split are matched. The previous results remain in the historical appendix under their original definitions.',[
    'configs/professor_model/nov2_followup.yaml; results/metrics/professor_model/nov2_followup/run_manifest.json.',
    `${REFERENCE}; supplied 14-page reference PDF SHA-256 ${referenceHash}.`
  ]);
}
// 2. Rebuild the model diagram as editable objects, correcting score notation.
{
  const s=slide('From True Risk to Lending Decisions','Working contract: T=30, theta=0.10, r=5% per period. Fixed capital $1.826087B.');
  box(s,'Synthetic applicants\nY, D, B, C',56,154,330,100);arrow(s,400,197,36,13);
  box(s,'Hidden true default risk\ntrue_PD = f(Y, D, ε)',456,154,330,100);arrow(s,800,197,36,13);
  box(s,'Noisy target PN\nPN = 0.8 true_PD + η',856,154,368,100,C.maroon);arrow(s,1032,263,16,44,'down');
  text(s,'Scenario 1 predictors: Y, D, C\nScenario 2 predictors: Y, D, C, B',56,328,570,84,28);
  box(s,'Bank estimates raw scores\nOLS / OLS + 0.01B / gradient boosting',664,318,560,110,C.maroon,26);arrow(s,1032,439,16,48,'down');
  box(s,'Capital-constrained\ncommon lending cutoff',56,503,330,94);arrow(s,401,543,36,13,'left');
  box(s,'Expected loan profit\nΣ c_t (1−p)^t − L',456,503,330,94);arrow(s,800,543,36,13,'left');
  box(s,'Accounting probability\np = clip(raw, 0, 1)',856,503,368,94);
  text(s,'Approval decisions, group outcomes, and evaluator-only true expected profit',56,630,1168,40,27,C.maroon,true);
  note(s,'The overall model is unchanged in its economic accounting. The bank observes applicant characteristics and learns the noisy PN target, but it does not receive true PD or epsilon as predictors. The three methods produce raw scores. I then clip those scores to probability bounds, calculate expected loan profit, and optimize the same common bounded-PD cutoff under capital. The penalty enters the raw score before clipping. PN is the target, not the fitted prediction. Higher estimated default risk disfavors an applicant. True default probability means constant conditional per-period default risk, with absorbing default.',[
    'src/fair_lending/professor_model/risk.py and profit.py; frozen true-risk and survival equations.',
    'configs/professor_model/loan_economics.yaml: c_t denotes scheduled payment ell*(1+r)^t, principal T*ell, no funding costs, recovery or discounting.',
    'The overview shows the common workflow, not an unrestricted portfolio optimizer or a calibrated mortgage model.'
  ]);
}
// 3. No systemic-world setup.
{
  const s=slide('Scenario 1: No Systemic Differences','Original group-neutral population. Same applicants, saved noise and fixed evaluation cohort.');
  text(s,'B=0 and B=1 share the same underlying risk process',56,176,1168,130,47,C.maroon,true);
  text(s,'The bank fits PN using income, DTI and the credit signal.',56,344,1168,70,31);
  text(s,'Only the manual-penalty branch explicitly adds +0.01B.',56,447,1168,78,31);
  text(s,'Finite-sample group differences can still appear without a group effect in true risk.',56,585,1168,66,25,C.gray);
  note(s,'Scenario 1 uses the saved original population without changing epsilon, C, PN, contracts or split assignments. Group membership has no structural effect on the original true-risk process. Both OLS and gradient boosting use Y, D and C, excluding B. The penalty branch deliberately adds B after the OLS fit. Financial characteristics need not be exactly equal in the finite realized groups, but they were generated from the same group-neutral process.',[
    'data/synthetic/professor_model/baseline/; original saved applicants, private factors, PN, eta and applicant splits.',
    'Scenario 1 OLS control replays the frozen group-blind model, with no refit.'
  ]);
}
function methodsSlide(scenario,number) {
  const s=slide(`Scenario ${number}: Methods and Corrected Penalty`,'PN remains the continuous regression target. Raw scores and bounded accounting PD remain separate.');
  const features=number===1?'Y, D, C':'Y, D, C, B';
  text(s,'Higher estimated default risk disfavors B=1',56,151,1168,47,31,C.maroon,true);
  table(s,[['Method','Raw score','Information / adjustment'],
    ['OLS control',`f_OLS(${features})`,'Unadjusted fitted score'],
    ['OLS + penalty',`f_OLS(${features}) + 0.01 B`,'Same OLS fit, +1 pp for B=1'],
    ['Gradient boosting',`f_ML(${features})`,'Same predictors, no added penalty']],56,224,1168,300,[280,440,448],24);
  text(s,'The control and penalty share exactly the same OLS fit',56,557,1168,48,31,C.maroon,true);
  text(s,'ML settings use training-only validation PN error. Evaluation truth stays hidden.',56,618,1168,45,24,C.gray);
  note(s,`For Scenario ${number}, all fitted methods have the same allowed information set, ${features}. The OLS control and manual-penalty model are not two separate regressions: the penalty adds exactly 0.01 times B to the control raw score before clipping. Gradient boosting learns the continuous PN measurement with no added penalty. I select its four predeclared candidate settings using validation PN mean squared error inside the original training cohort, then refit the chosen setting on all 80,000 training applicants. Evaluation PN, true risk and profit are not used to choose settings. ${number===2?'Because B enters the Scenario 2 fits, I use the neutral label OLS control rather than calling this branch blind.':''}`, [
    'configs/professor_model/nov2_followup.yaml; results/metrics/professor_model/nov2_followup/model_metadata.json.',
    'HistGradientBoostingRegressor, squared-error loss; max_leaf_nodes {3,7}, max_iter {25,75}; learning_rate .05, min_samples_leaf 1000, l2_regularization 10, early_stopping False, random_state 4994.',
    'Common 64,000/16,000 fitting/validation split within the original 80,000 training IDs; PCG64 SeedSequence(4994,spawn_key=(1102,0)).',
    'The ML method is a regressor because PN is continuous and may be outside [0,1]. No binary default labels are fabricated.'
  ]);
}
methodsSlide(scenarios[0],1);
resultSlide(scenarios[0],1);
// 6. Systemic-world setup, matched primitives and transparent researcher choice.
{
  const s=slide('Scenario 2: Systemic Differences','Researcher-chosen +1 SD epsilon shift for B=1. Frozen true-risk coefficients remain unchanged.');
  text(s,'B=1 has a higher hidden risk factor',56,166,1168,73,43,C.maroon,true);
  text(s,'ε₂ = ε₁ + B',56,271,1168,58,39);
  text(s,'C₂ = 0.4 ε₂ + ω\nPN₂ = 0.8 true_PD₂ + the same η',56,360,1168,98,32);
  text(s,'OLS and ML now use Y, D, C and B',56,498,1168,57,33,C.maroon,true);
  const baselineB=systemicAudit.find(r=>r.scenario_id===scenarios[0]&&Number(r.B)===1);
  const shiftedB=systemicAudit.find(r=>r.scenario_id===scenarios[1]&&Number(r.B)===1);
  if(!baselineB||!shiftedB) throw new Error('Missing systemic population audit');
  text(s,`B=1 population mean true PD: ${f(100*num(baselineB,'mean_true_PD'),5)}% to ${f(100*num(shiftedB,'mean_true_PD'),5)}%.\nBetween-world results also reflect the change in permitted predictors.`,56,588,1168,78,24,C.gray);
  note(s,'Scenario 2 shifts epsilon upward by one standard deviation for B=1. I keep the same residual draws and recompute C so the credit-signal equation remains internally consistent. I then recalculate true PD using the original coefficients and rebuild PN using the same eta. Income, DTI, group identities, contracts, capital and split stay fixed. B=0 is unchanged. The bank still cannot observe epsilon directly, although C partially signals it. OLS and ML now include B, as the revised slide specifies. The existing risk sigmoid is near saturation, so a one-standard-deviation epsilon shift does not imply a large default-probability gap. I do not retune it to produce an expected graph.',[
    'configs/professor_model/nov2_followup.yaml: systemic_epsilon_shift=1.0; scenario-specific feature allowlists.',
    `Original true PD = sigmoid(.52-.65 logY)*sigmoid(4+.06D+epsilon). No extra B term is inserted into that equation. Population B=1 mean PD changes from ${num(baselineB,'mean_true_PD')} to ${num(shiftedB,'mean_true_PD')} raw probability units.`,
    'This is an explicitly researcher-defined synthetic upstream difference. The between-scenario comparison changes both risk conditions and whether B is permitted in the fit; it is not an isolated epsilon-only effect.'
  ]);
}
methodsSlide(scenarios[1],2);
resultSlide(scenarios[1],2);
// 9-14. Full original pages, retained as historical evidence, not new results.
const historical=[
  ['Moving the Tie Leaves Lending Unchanged','The original -0.1B treatment and positive-floor sensitivity.'],
  ['Removing B’s Contribution Without Refitting','The prior W/Z/F coefficient-removal comparison.'],
  ['No Universal Profit Gain Across 200 Draws','The prior 200-noise-draw study on one fixed population.'],
  ['Ranking Scores and Accounting Probabilities','The separate prior raw-ordering policy sensitivity.'],
  ['Nine Reject-All Policies Now Fund Loans','The prior six selected raw-ordering diagnostic cases.'],
  ['Decisions Before the Final Research Report','Historical discussion questions, not a current work-status claim.'],
];
for(let page=9;page<=14;page++) {
  const s=P.slides.add();s.background.fill=C.white;
  s.shapes.add({geometry:'rect',position:{left:0,top:0,width:1280,height:40},fill:C.maroon,line:{fill:'none',width:0}});
  text(s,'Historical appendix: earlier specification and saved results',40,8,1110,27,20,C.white,true);
  text(s,String(page).padStart(2,'0'),1192,8,48,27,20,C.white,false,'right');
  const image=path.join(BUILD,`reference-${String(page).padStart(2,'0')}.png`);
  s.images.add({blob:new Uint8Array(await fs.readFile(image)),contentType:'image/png',fit:'contain',position:{left:0,top:40,width:1280,height:680},alt:`Historical original source page ${page}: ${historical[page-9][0]}`});
  note(s,`This page preserves the earlier slide for context. ${historical[page-9][1]} Its numbers, methods, and original page marker are unchanged. It does not describe the new +0.01B/Systemic/ML run. The new November 2 conclusions are on slides 3 through 8.`,[
    `${REFERENCE}, original PDF page ${page}. Source SHA-256 ${referenceHash}.`,
    'Historical reference image: full page preserved with proportional fit, below an explicit historical banner. Native new-result charts appear only on slides 5 and 8.',
    page===14?'The prior question about adding an ML benchmark is historical; this new experiment now implements the specified gradient-boosting regressor.':'The historical experiment was not rerun or reinterpreted as a new result.'
  ]);
}
if(P.slides.items.length!==14) throw new Error('Expected exactly fourteen slides');
await fs.writeFile(path.join(BUILD,`${STEM}.evidence.json`),JSON.stringify({reference:REFERENCE,reference_sha256:referenceHash,inputs,columns:col,groups,portfolios,diagnostics,comparisons,systemicAudit,validation},null,2));
await fs.writeFile(path.join(BUILD,`${STEM}.speaker_notes.json`),JSON.stringify(speaking,null,2));
await fs.writeFile(path.join(BUILD,`${STEM}.inspection.ndjson`),(await P.inspect({kind:'slide,textbox,table,chart,notes',maxChars:200000})).ndjson);
const candidate=path.join(BUILD,`${STEM}.candidate.pptx`);
await (await PresentationFile.exportPptx(P)).save(candidate);
await finalizePresentation({workspaceDir:WORKSPACE,candidatePath:candidate,finalPath:outPptx,
  pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  explicitTotalSlideCount:14,requiredNativeTableOwnerSlides:[4,7],requiredNativeChartOwnerSlides:[5,8],
  materializeLiteralChartWorkbooks:true,verifyArtifactToolImport:true,
  fontPolicy:{basis:'reference',families:[font],referencePath:REFERENCE,referenceSha256:referenceHash},
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',... [4,7].flatMap(i=>['--require-native-table-slide',String(i)])],
  receiptPath:path.join(BUILD,`${STEM}.validation.json`)});
const final=await PresentationFile.importPptx(await FileBlob.load(outPptx));
const previews=path.join(BUILD,`previews_${REV}`);await fs.mkdir(previews,{recursive:true});
for(let i=0;i<final.slides.items.length;i++) {
  const blob=await final.export({slide:final.slides.items[i],format:'png',scale:1.5});
  await fs.writeFile(path.join(previews,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
if(sha(await fs.readFile(REFERENCE))!==referenceHash) throw new Error('Reference PDF changed');
await fs.writeFile(path.join(BUILD,`${STEM}.rendering.json`),JSON.stringify({slides:14,native_charts:4,native_tables:2,editable_overview:true,historical_images:6,previews,outPptx,reference_sha256:referenceHash,powerpoint_open_verified:false},null,2));
console.log(JSON.stringify({outPptx,previews,slides:14}));
