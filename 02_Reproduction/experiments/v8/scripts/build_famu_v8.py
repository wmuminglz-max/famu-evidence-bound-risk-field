"""Freeze scoped synthetic evidence supervision and unchanged V4 replay, V8."""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evidence_reasoning.famu_v8 import FIELDS, compose
OUT=ROOT/'training_data/famu_v8_20261005'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in p.read_text('utf-8').splitlines() if s]
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')

def scene(split, n, style, axis, variant, states, rule):
    serial={'train':1000,'dev':2000,'test':3000}[split]+n
    actor=f'澄岳{serial}号公司';other=f'澄岳{serial+1}号公司'
    event=['交付检测仪','提交复核报告','返还借用设备'][n%3]
    year=2027;month=n%8+1;deadline=f'{year}年{month}月12日';before=f'{year}年{month}月8日';after=f'{year}年{month}月18日'
    scope={'actor':actor,'event':event,'matter':f'项目{serial}本次履行事项','time':f'{year}年{month}月',
           'source':'本事项完整签署原件及本期履行记录；展示脱敏件不等于原件','assessment_deadline':deadline}
    fields={'A':f'本事项完整签署原件载明{actor}须于{deadline}前{event}',
            'B':f'{actor}本事项实际在{deadline}之后才{event}',
            'C':f'{actor}已获准免除本事项此次逾期处理',
            'D':f'{actor}本事项本期结算仍有未支付余额'}
    lead=['本案核验记录：','签章档案逐项记载：','本期原始凭据内容：','经对照的案内记录为：','从双方认可的本期原件读取到：','当前事项的原始凭证注明：'][style]
    a={
      'T':[f'{lead}完整签署原件第七条约定：{actor}应在{deadline}前{event}。',f'{lead}原件完整可见，{actor}{event}的最后期限列为{deadline}。'],
      'F':[f'{lead}已逐页核对本事项全部签署原件与附件，{actor}{event}没有任何截止日期约定，期限栏原样空白。',f'{lead}全套原件完整，关于{actor}{event}的时限未填写，其他条文亦没有补充；并非脱敏或漏扫。'],
      'U':[f'{lead}展示副本写“{actor}须于[日期已隐去]前{event}”，未取得签署原件，无法核实原稿是否填写。',f'{lead}只有{actor}{event}条款的OCR残页，日期识别失败；完整签署原件尚未调取。']}
    b={
      'T':[f'{lead}{actor}本事项双方签认的实际回执记载：{after}才{event}，此前未完成。',f'{lead}{actor}在本事项中的实际办结日为{after}，办结内容是{event}；双方已确认此日。'],
      'F':[f'{lead}{actor}本事项双方签认的实际回执记载：{before}已全部{event}，没有剩余部分。',f'{lead}{actor}本事项已于{before}办结{event}，实际完成情况由双方签认，并非计划。'],
      'U':[f'{lead}{actor}承诺按时{event}，但未提供本事项任何实际完成记录；现有文件仅是履行计划。',f'{lead}仅有{actor}本事项的计划安排，不能查得实际{event}时间，没有实绩凭证。']}
    c={
      'T':[f'{lead}双方已签署生效的本次豁免确认单，明确准许{actor}免于本事项此次逾期处理，无其他条件。',f'{lead}针对{actor}本次事项，双方已确认逾期处理全部免予执行；该确认不是未批申请。'],
      'F':[f'{lead}双方已签署本事项本次决定：“{actor}不得免于此次逾期处理，本次不适用任何豁免。”',f'{lead}关于{actor}本事项的适用决定明确禁止本次豁免；双方已确认，不存在其他例外许可。'],
      'U':[f'{lead}{actor}提交了本事项逾期豁免申请，尚未提供任何批准或驳回决定。',f'{lead}本次仅见{actor}请求免除处理的申请，未收到适用决定，不能确定准许还是禁止。']}
    d={
      'T':[f'{lead}{actor}本事项本期双方核验台账：应付420元，实付120元，尚欠300元。',f'{lead}{actor}本事项本期仍有300元未付余额，已核对实际到账流水。'],
      'F':[f'{lead}{actor}本事项本期双方核验台账：应付420元，实付420元，余额0元，款项已结清。',f'{lead}{actor}本事项本期逐笔对账完毕，应付费用已全部支付，没有剩余欠款。'],
      'U':[f'{lead}{actor}本事项本期仅有应付420元的计划，实际付款流水未提供，支付情况未知。',f'{lead}{actor}本事项的实付金额在展示件中已脱敏，原始台账缺失，无法核对余额。']}
    texts=[a,b,c,d]; facts=[]
    for f,state,table in zip(FIELDS,states,texts):
        text=table[state][(n+style)%2]
        if f=='B' and state=='U' and axis=='actor':text=f'{lead}{other}本事项已于{after}{event}。{actor}本事项的实际履行记录未提供。'
        if f=='B' and state=='U' and axis=='time':text=f'{lead}{actor}上一年度另一事项的回执记为2026年{month}月18日{event}。2027年本事项的实际记录未提供。'
        facts.append({'text':text,'field':f,'state':state})
    # Long distractors are scoped to distinct actors/years. Positions and IDs never encode the label.
    count=([12,32,64] if split!='test' else [32,64,96])[n%3]
    for i in range(count):
        xactor=actor if i%3==0 else f'澄岳{serial+i+20}号公司';xyear=2022+i%4 if i%3==0 else 2027
        fragment=[f'{xactor}在{xyear}年的另一检测事项中约定于5月12日交付，回执显示5月18日完成，处理申请未获豁免。',
                  f'{xactor}另一事项的全套合同没有填写交付期限，双方实际结算已经付清；该页为{xyear}年归档件。',
                  f'{xyear}年{xactor}的复核报告提交申请中载有计划日期，实际履行日期未查；这并非本次事项的原件。',
                  f'{xactor}上一独立项目在{xyear}年得到免于处理的批准，当期尚欠三百元，该批准仅适用于该独立项目。'][i%4]
        facts.append({'text':lead+fragment,'field':None,'state':None})
    rng=random.Random(serial*43);rng.shuffle(facts)
    evidence=[{'id':f'P{i+1:03d}','text':x['text']} for i,x in enumerate(facts)]
    bindings={f:{'s':[],'r':[]} for f in FIELDS}
    for e,x in zip(evidence,facts):
        if x['field'] and x['state'] in 'TF':bindings[x['field']]['s' if x['state']=='T' else 'r'].append(e['id'])
    return {'id':f'V8-{split}-{n:03d}-{axis}-{variant}', 'family':f'V8-{split}-{n:03d}',
            'pair_group':f'V8-{split}-{n:03d}-{axis}', 'split':split,'axis':axis,'style':style,
            'scope':scope,'fields':fields,'rule':rule,'evidence':evidence,'states':states,
            'bindings':bindings,'target':compose(rule,states),
            'label_authority':'new_fictional_program_supervision_not_expert_legal_gold'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    and_rule={'all':['A','B',{'not':'C'}]}
    for split,count,styles in [('train',18,[0,1,2]),('dev',4,[3]),('test',6,[4,5])]:
        for n in range(count):
            for axis in ['source','event','exception','payment','actor','time']:
                for idx,s in enumerate('TFU'):
                    states=['T','T','F','F'];rule=and_rule
                    if axis=='source':states[0]=s;rule={'not':'A'}
                    elif axis in ['event','actor','time']:states[1]=s
                    elif axis=='exception':states[2]=s
                    else:states[3]=s;rule='D'
                    rows.append(scene(split,n,styles[n%len(styles)],axis,idx,states,rule))
            specs=[(['T','U','T','F'],and_rule),(['F','U','U','F'],and_rule),(['T','F','U','F'],and_rule),
                   (['T','T','U','F'],and_rule),(['T','U','F','T'],{'any':[and_rule,'D']}),
                   (['T','F','F','U'],{'any':[and_rule,'D']})]
            for idx,(states,rule) in enumerate(specs):rows.append(scene(split,n,styles[n%len(styles)],'composition',idx,states,rule))
    v4=ROOT/'training_data/famu_v4_20261003'
    replay=read(v4/'gold_train.jsonl');regression=read(v4/'internal_dev.jsonl')
    assert len(replay)==77 and len(regression)==9
    for name,data in [('scenes.jsonl',rows),('replay77.jsonl',replay),('regression9.jsonl',regression)]:
        p=OUT/name;content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data)
        if p.exists():assert p.read_text('utf-8')==content,'Refuse changing frozen data'
        else:p.write_text(content,'utf-8')
    assert len({r['id'] for r in rows})==len(rows)
    assert not {r['family'] for r in rows if r['split']=='train'} & {r['family'] for r in rows if r['split']!='train'}
    manifest={'version':'V8-evidence-bound-1','counts':dict(Counter(r['split'] for r in rows)),
        'labels':dict(Counter(r['split']+'/'+r['target'] for r in rows)),
        'files':{n:sha(OUT/n) for n in ['scenes.jsonl','replay77.jsonl','regression9.jsonl']},
        'source_v4':{n:sha(v4/n) for n in ['gold_train.jsonl','internal_dev.jsonl','manifest.json']},
        'replay77_unchanged':True,'historical_boundary54_used_for_training':False,'LAB9_training':False,
        'independent_legal_test':False,'synthetic_wrapper_holdouts':{'train':[0,1,2],'dev':[3],'test':[4,5]},
        'shared_sentence_templates_across_splits':True,
        'synthetic_exposures_each':3,'replay_exposures_each':4,'queries_per_run':1604,'updates_per_run':401,
        'arms':['D0_direct','E1_bound'],'seeds':[17,29,43],'total_updates':2406,
        'matched_cost':'same inputs/example exposure/update count; not equal output tokens or wall time'}
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text('utf-8'))==manifest
    else:write(OUT/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
