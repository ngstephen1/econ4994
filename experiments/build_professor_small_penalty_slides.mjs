/**
 * Build the 14-slide professor follow-up from already validated saved results.
 * This script never generates applicants, fits models, or selects loans.
 *
 * Run with the bundled Node runtime. Override RUNTIME_NODE_MODULES,
 * PRESENTATIONS_SKILL_DIR, RUNTIME_PYTHON, or REFERENCE_PDF when needed.
 * Each SMALL_PENALTY_SLIDE_REVISION writes a separate immutable candidate/delivery.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const WORKSPACE = path.join(ROOT, 'reports/professor_model/small_penalty');
const BUILD = path.join(WORKSPACE, '_build');
const DELIVERY = path.join(WORKSPACE, 'delivery');
const REV = process.env.SMALL_PENALTY_SLIDE_REVISION ?? '01';
if (!/^\d{2}$/.test(REV)) throw new Error('SMALL_PENALTY_SLIDE_REVISION must contain two digits');
const STEM = `professor_small_penalty_${REV}`;
const REFERENCE = process.env.REFERENCE_PDF ?? '/Users/macbook/Desktop/nov_2_followup_slide_update.pptx.pdf';
const MODULES = process.env.RUNTIME_NODE_MODULES ?? '/Users/macbook/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE_MODULES ??= MODULES;
const SKILL = process.env.PRESENTATIONS_SKILL_DIR ?? '/Users/macbook/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const PY = process.env.RUNTIME_PYTHON ?? '/Users/macbook/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const {Presentation, PresentationFile, FileBlob} = await import(pathToFileURL(path.join(MODULES, '@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {finalizePresentation, resolvePresentationFont, applyPresentationChartFont} = await import(pathToFileURL(path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')).href);
const TABLES = path.join(ROOT, 'results/tables/professor_model/small_penalty');
const METRICS = path.join(ROOT, 'results/metrics/professor_model/small_penalty');
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

function command(executable, args) {
  const result = spawnSync(executable, args, {encoding:'utf8', maxBuffer:30e6});
  if (result.status !== 0) throw new Error(`${executable} failed: ${result.stderr || result.stdout}`);
  return result.stdout;
}
function readCSV(filename) {
  return JSON.parse(command(PY, ['-c', 'import csv,json,sys; print(json.dumps(list(csv.DictReader(open(sys.argv[1],newline="")))))', filename]));
}
const requiredFiles = ['group_results.csv', 'portfolio_results.csv', 'risk_diagnostics.csv', 'penalty_comparisons.csv', 'penalty_search.csv', 'systemic_risk_audit.csv'];
const inputs = {};
for (const name of requiredFiles) inputs[path.relative(ROOT,path.join(TABLES,name))] = sha(await fs.readFile(path.join(TABLES,name)));
for (const name of ['run_manifest.json','validation.json','bank_freeze_checkpoint.json','bank_selections.json','selection.json']) {
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
const selected = JSON.parse(await fs.readFile(path.join(METRICS,'selection.json'),'utf8'));
const delta = Number(selected.selected_delta);
if(selected.status!=='criterion_met'||!Number.isFinite(delta)||delta<=0) throw new Error('Selected common positive penalty is required');
const deltaText=String(delta);
const deltaPP=String(Number((delta*100).toFixed(8)));

if(validation.status!=='validated' || runManifest.status!=='complete_small_penalty_prediction_only') {
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
  if(m==='ols_penalty' && !(num(b,col.count)>0 && num(b,col.count)*num(a,col.n)<num(a,col.count)*num(b,col.n))) throw new Error('Chosen penalty fails positive-but-lower criterion');
}

await fs.mkdir(BUILD,{recursive:true});
await fs.mkdir(DELIVERY,{recursive:true});
const outPptx=path.join(DELIVERY,`${STEM}.pptx`);
for(const existing of [outPptx,path.join(BUILD,`${STEM}.validation.json`)]) {
  try { await fs.access(existing); throw new Error(`Refusing to overwrite existing revision: ${existing}`); }
  catch(error) { if(error.code!=='ENOENT') throw error; }
}
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
const minPositiveProfit=Math.min(...allProfit.filter(v=>v>0));
const profitDigits=Math.max(4,Math.min(8,Math.ceil(-Math.log10(minPositiveProfit))+1));
const profitNumberFormat='0.'+'0'.repeat(profitDigits);
function resultChart(s,scenario,kind,x) {
  const approval=kind==='approval';
  const series=[0,1].map(b=>({name:b===0?'Group A':'Group B',fill:b===0?C.slate:C.orange,values:methods.map(m=>cv(approval?num(group(scenario,m,b),col.rate):num(group(scenario,m,b),col.groupProfit)/1e9))}));
  const chart=s.charts.add('bar',{
    position:{left:x,top:235,width:568,height:354},categories:labels,series,
    hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{fontSize:20,fill:C.slate}},
    barOptions:{direction:'column',grouping:'clustered',gapWidth:40,overlap:0},
    xAxis:{...axis,tickLabelPosition:'low'},
    yAxis:{...axis,min:approval?0:profitMin,max:approval?1:profitMax,majorUnit:approval?.2:.5,numberFormatCode:approval?'0%':'0.0',majorGridlines:{fill:C.line,width:.7}},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:approval?'0.00%':profitNumberFormat,textStyle:{fontSize:17.5,fill:C.slate,bold:true}},
    chartFill:C.white,plotAreaFill:C.white,
  });
  applyPresentationChartFont(chart,{fontFamily:font});
  return chart;
}
function resultSlide(scenario,number) {
  const s=slide(`Scenario ${number}: Prediction-Subset Results`,'Source: small_penalty/group_results.csv and portfolio_results.csv. Same contracts and capital.');
  const mlUtilization=num(portfolio(scenario,'ml_histgb'),'capital_utilization');
  text(s,`Profits: prediction subset only. Gradient boosting uses ${f(100*mlUtilization,1)}% of capital.`,56,147,1168,37,25);
  text(s,'Within-group approval rate',56,195,568,40,27,C.slate,true);
  text(s,'True expected profit by group ($B)',656,195,568,40,27,C.slate,true);
  resultChart(s,scenario,'approval',56);resultChart(s,scenario,'profit',656);
  const control=portfolio(scenario,'ols_control'),penalty=portfolio(scenario,'ols_penalty');
  const a=group(scenario,'ols_penalty',0),b=group(scenario,'ols_penalty',1);
  text(s,`Penalty +${deltaText}B: A ${f(num(a,col.count),0)}/${f(num(a,col.n),0)} (${f(100*num(a,col.rate),2)}%); B ${f(num(b,col.count),0)}/${f(num(b,col.n),0)} (${f(100*num(b,col.rate),2)}%).`,56,602,1168,33,23,C.maroon,true);
  text(s,`Penalty minus control: true expected portfolio profit ${dollars(num(penalty,col.profit)-num(control,col.profit))}`,56,642,1168,31,23,C.maroon,true);
  note(s,`All results here use the same 20,000 prediction applicants, never the 80,000 training applicants. For Scenario ${number}, the selected common penalty is ${deltaText} times B, added to the frozen OLS raw score before clipping. Group A has ${num(a,col.count)} approvals out of ${num(a,col.n)} and minority Group B has ${num(b,col.count)} out of ${num(b,col.n)}. Group B therefore has positive lending at a lower rate than A. The true expected portfolio profit difference from control is ${dollars(num(penalty,col.profit)-num(control,col.profit))}. The profit bars sum each group's funded-loan contributions, so unequal group sizes matter. ML uses ${f(100*mlUtilization,4)} percent of capital; its next indivisible prediction block cannot fit under the common cutoff. More approvals need not mean greater capital use or greater profit. These are nominal synthetic expected values, not annual realized returns.`,[
    'results/tables/professor_model/small_penalty/group_results.csv; portfolio_results.csv; penalty_comparisons.csv.',
    'Raw model predictions, control/ML portfolios, the 80,000/20,000 split, contracts and capital are reused unchanged from the November 2 experiment.',
    'Penalty selection uses prediction-cohort group approval counts for illustration, not hidden true risk or true profit. The selected comparison is not an untouched confirmatory test.',
    'Group A = B=0. Minority Group B = B=1. Exact approval counts remain visible because a small positive rate can otherwise look like zero.'
  ]);
}

// 1. Final update framed around the professor's research interpretation.
{
  const s=P.slides.add();s.background.fill=C.white;
  text(s,'Lending Under Imperfect Information',56,74,1168,88,54,C.maroon,true);
  text(s,'Final update: discrimination and predictive lending',56,185,1168,66,32);
  text(s,'Nguyen Phan Nguyen\nVirginia Tech - ECON 4994\n3 November 2026',56,281,1000,108,24,C.gray);
  text(s,`How does direct discrimination affect bank profit,\nand how does gradient boosting compare with OLS?`,56,433,1168,111,33,C.maroon,true);
  text(s,`Group A (B=0). Minority Group B (B=1). Selected penalty: +${deltaText}B.`,56,570,1168,45,25);
  text(s,'80,000 training. 20,000 prediction. Profits only from the prediction subset.',56,627,1168,41,24,C.slate,true);
  text(s,'Single matched synthetic run. Penalty calibrated to prediction-group approvals for illustration.',56,682,1134,27,14,C.gray);
  text(s,'01',1208,682,32,24,14,C.gray,false,'right');
  note(s,`Following your feedback, I reduced the manual penalty. I started with 0.005 times B and used the predeclared descending sequence only if needed. The common selected value is ${deltaText}. Group B now receives some loans but has a lower approval rate than Group A in both risk scenarios. The models remain frozen from the previous run. They learned from 80,000 training applicants, and all lending and profit results use the separate 20,000 prediction applicants. I frame the comparison as naïve OLS estimation, direct discrimination through an added penalty, and a more flexible gradient-boosting estimator, without assuming that ML must win.`,[
    'Professor feedback supplied in the conversation; the 14-slide structure and earlier results remain preserved in prior deliveries.',
    'configs/professor_model/small_penalty.yaml; results/metrics/professor_model/small_penalty/selection.json; run_manifest.json.',
    `${REFERENCE}; supplied source PDF SHA-256 ${referenceHash}.`,
    'The penalty is chosen using prediction-cohort approval counts. This is an illustrative policy calibration, not independent empirical evidence of discrimination.'
  ]);
}
// 2. Explicit learning/prediction stages and the unchanged economic model.
{
  const s=slide('Training, Prediction and Bank Profit','Working contract: T=30, theta=0.10, r=5% per period. Fixed capital $1.826087B.');
  box(s,'80,000 TRAINING applicants\nObserved characteristics, credit proxy C\nand historical-style noisy PN',56,153,520,118,C.slate,24);
  arrow(s,601,200,68,16);
  box(s,'20,000 PREDICTION applicants\nFrozen OLS and gradient-boosting models\nproduce default-risk scores',704,153,520,118,C.maroon,24);
  text(s,'The bank cannot observe true PD or every risk determinant.',56,300,1168,47,28,C.maroon,true);
  text(s,'PN = 0.8 true_PD + η\nC is a noisy proxy for the hidden risk factor.',56,367,570,90,27);
  box(s,`OLS control / OLS + ${deltaText}B / ML\nBounded PD = clip(raw score, 0, 1)`,664,361,560,99,C.maroon,25);
  arrow(s,1032,470,16,41,'down');
  box(s,'Capital-constrained\ncommon lending cutoff',56,514,330,91);
  arrow(s,401,553,36,13,'left');
  box(s,'Expected loan profit\nΣ c_t (1−p)^t − L',456,514,330,91);
  arrow(s,800,553,36,13,'left');
  box(s,'Prediction-subset\nexpected cash flows',856,514,368,91);
  text(s,'All approvals and all reported profits use prediction applicants only',56,635,1168,42,29,C.maroon,true);
  note(s,'This makes the split explicit. The training subset mimics past loan information and prior noisy risk measurements. OLS and Gradient Boosting use permitted characteristics and the credit proxy to learn PN. The bank does not observe true PD or every determinant of risk. We then apply frozen models to a separate prediction subset, add the manual group penalty only in that branch, clip scores for economic accounting, and select a common cutoff under capital. We calculate profits only from those prediction applicants. For ML, its original 64,000/16,000 fitting-validation subdivision is entirely within the 80,000 training applicants, followed by an 80,000-row refit. No model was refit for this smaller-penalty update.',[
    'Original applicant_splits.csv and model_metadata.json from the frozen November 2 experiment; new split audit verifies disjointness and exact prediction-contract coverage.',
    'The original contracts/profit functions use constant conditional per-period default and a survival-weighted payment sum. They are stylized expected values, not conventional mortgage returns.',
    'No prediction target, hidden true risk or true profit is used to fit or choose the original models.'
  ]);
}
// 3. Scenario 1 setup.
{
  const s=slide('Scenario 1: No Systemic Differences','Same group-neutral generating process. Finite realized group averages may differ.');
  text(s,'Group A and minority Group B share\nthe same underlying risk process',56,170,1168,137,44,C.maroon,true);
  text(s,'The bank observes Y, D and C to estimate default risk.',56,340,1168,70,31);
  text(s,`The direct-discrimination branch adds +${deltaText}B\nafter the OLS fit and before probability clipping.`,56,437,1168,100,31);
  text(s,'The added group penalty is arbitrary relative to the group-neutral true-risk process.',56,584,1168,65,25,C.gray);
  note(s,'In the first risk scenario, Group A and minority Group B are generated under the same underlying risk conditions. This does not require their finite sample means to be numerically identical. Neither the OLS control nor Gradient Boosting includes B. The bank with direct discrimination adds a positive penalty after estimation, despite the lack of an underlying group treatment in true risk. This lets us compare an unadjusted estimate with the same estimate plus arbitrary discriminatory treatment.',[
    'Original November 2 Scenario 1 population, truth, observed inputs and noise remain unchanged.',
    'Scenario 1 feature set: Y, D, C for both OLS and Gradient Boosting. The only new score change is the selected post-fit penalty.'
  ]);
}
function methodsSlide(scenario,number) {
  const s=slide(`Scenario ${number}: Three Lending Methods`,'Same permitted information within each scenario. Historical models and prediction scores remain frozen.');
  const features=number===1?'Y, D, C':'Y, D, C, B';
  text(s,'Higher estimated default risk lowers lending priority',56,151,1168,47,31,C.maroon,true);
  table(s,[['Method','Raw score','Interpretation'],
    ['OLS control',`f_OLS(${features})`,'Naïve risk estimation'],
    ['OLS + penalty',`f_OLS(${features}) + ${deltaText} B`,`Direct discrimination: +${deltaPP} pp for B=1`],
    ['Gradient boosting',`f_ML(${features})`,'Flexible predictive estimation']],56,224,1168,300,[280,440,448],23);
  text(s,'Training: 80,000. Prediction and profit: 20,000.',56,556,1168,45,30,C.maroon,true);
  text(s,`Penalty selected from prediction-group approvals for illustration. True profit is not a selection criterion.`,56,616,1168,52,23,C.gray);
  note(s,`These are three methods under Scenario ${number}, not three different applicant populations. OLS control is the naïve linear risk estimator. The discriminatory method uses exactly the same fitted OLS score and adds ${deltaText} times B. Gradient Boosting is the flexible predictive comparator, with performance evaluated empirically rather than assumed to be superior. The information set is ${features}; ${number===2?'B is included in the second scenario because group membership can contain information about the systemic difference. It is therefore not a group-blind fit.':'B is excluded from the first scenario fitted models.'} The common penalty was selected using prediction-cohort approval rates to meet the professor’s illustrative condition. This does not contaminate the model-fitting split with prediction labels, but the policy choice is not an untouched confirmatory evaluation.`,[
    'results/metrics/professor_model/small_penalty/selection.json; results/tables/professor_model/small_penalty/penalty_search.csv.',
    'Parent November 2 model metadata: original 80,000 training rows, 64,000/16,000 internal ML selection, then final 80,000-row refit.',
    'No OLS refit, ML tuning, new noise, new population or true-profit-based penalty choice occurs in this update.'
  ]);
}
methodsSlide(scenarios[0],1);
resultSlide(scenarios[0],1);
// 6. Existing systemic world.
{
  const s=slide('Scenario 2: Systemic Differences','Existing +1 SD epsilon shift for minority Group B. No new population or risk-equation change.');
  text(s,'Group B has a higher hidden risk factor\nfrom a process outside the lending decision',56,166,1168,116,40,C.maroon,true);
  text(s,'ε₂ = ε₁ + B',56,307,1168,58,37);
  text(s,'C₂ = 0.4 ε₂ + ω\nPN₂ = 0.8 true_PD₂ + the same η',56,389,1168,94,30);
  text(s,'OLS and Gradient Boosting use Y, D, C and B',56,517,1168,57,31,C.maroon,true);
  const baselineB=systemicAudit.find(r=>r.scenario_id===scenarios[0]&&Number(r.B)===1);
  const shiftedB=systemicAudit.find(r=>r.scenario_id===scenarios[1]&&Number(r.B)===1);
  if(!baselineB||!shiftedB) throw new Error('Missing existing systemic risk audit');
  text(s,`Group B mean true PD: ${f(100*num(baselineB,'mean_true_PD'),5)}% to ${f(100*num(shiftedB,'mean_true_PD'),5)}%.\nThe risk function is unchanged. Between-world results also change the permitted predictors.`,56,594,1168,72,23,C.gray);
  note(s,'The second scenario represents a systemic difference outside this bank’s lending decision, potentially analogous to unequal treatment in other domains or the past. We do not claim real-world evidence about why any actual group is riskier. The existing synthetic world shifts epsilon by one standard deviation for Group B and updates C consistently. The noisy measurement uses the same eta. That existing risk response is modest because the frozen sigmoid is near saturation. We preserve it without retuning. Both fitted methods include B in this scenario. Changing both risk conditions and the feature set means the cross-scenario contrast is not a pure epsilon-only causal effect.',[
    'results/tables/professor_model/small_penalty/systemic_risk_audit.csv is a hash-identical copy of the parent audit.',
    'Original risk: sigmoid(.52-.65 logY)*sigmoid(4+.06D+epsilon); C=.4epsilon+omega. The small-penalty study does not regenerate these variables.'
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
  note(s,`This page preserves the earlier slide for context. ${historical[page-9][1]} Its numbers, methods, and original page marker are unchanged. It does not describe the new smaller-penalty demonstration. The new smaller-penalty results are on slides 3 through 8.`,[
    `${REFERENCE}, original PDF page ${page}. Source SHA-256 ${referenceHash}.`,
    'Historical reference image: full page preserved with proportional fit, below an explicit historical banner. Native new-result charts appear only on slides 5 and 8.',
    page===14?'The prior question about adding an ML benchmark is historical; the parent November 2 experiment already implemented the frozen gradient-boosting regressor.':'The historical experiment was not rerun or reinterpreted as a new result.'
  ]);
}
if(P.slides.items.length!==14) throw new Error('Expected exactly fourteen slides');
await fs.writeFile(path.join(BUILD,`${STEM}.evidence.json`),JSON.stringify({reference:REFERENCE,reference_sha256:referenceHash,inputs,columns:col,groups,portfolios,diagnostics,comparisons,systemicAudit,validation,selected},null,2));
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
for(const [relative,digest] of Object.entries(inputs)) {
  if(sha(await fs.readFile(path.join(ROOT,relative)))!==digest) throw new Error(`Presentation input changed during authoring: ${relative}`);
}
await fs.writeFile(path.join(BUILD,`${STEM}.rendering.json`),JSON.stringify({slides:14,native_charts:4,native_tables:2,editable_overview:true,historical_images:6,previews,outPptx,reference_sha256:referenceHash,powerpoint_open_verified:false},null,2));
console.log(JSON.stringify({outPptx,previews,slides:14}));
