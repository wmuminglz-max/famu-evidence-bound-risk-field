"""Freeze R2 inputs and a newly authored synthetic challenge before training."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/famu_student_gpu_20261009'
OUT=ROOT/'outputs/famu_student_round2_20261010'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')


def truth(rule, states):
    def boolean(expr, assignment):
        if isinstance(expr,str):return assignment[expr]
        if 'not' in expr:return not boolean(expr['not'],assignment)
        key=next(iter(expr));values=[boolean(c,assignment) for c in expr[key]]
        return all(values) if key=='all' else any(values)
    outcomes=set()
    for values in itertools.product(*[[False,True] if s=='U' else [s=='T'] for s in states]):
        outcomes.add(boolean(rule,dict(zip('ABCD',values))))
    return 'unknown' if len(outcomes)>1 else 'established' if True in outcomes else 'not_established'


def challenge():
    result=[]
    for family in range(6):
        actor=['映川科技有限公司','远澜设备服务公司','启禾检验中心','新岚技术公司','和朔供应公司','北辰项目服务社'][family]
        event=['移交测量组件','提交校验清单','归还周转仪器'][family%3]
        month=family+2;due=f'2029年{month}月16日';before=f'2029年{month}月11日';after=f'2029年{month}月23日'
        scope={'actor':actor,'event':event,'matter':f'核验项目R{family}本次事项','time':f'2029年{month}月',
            'source':'本次签署全套原本与双方核对记录','assessment_deadline':due}
        fields={'A':f'本事项完整签署原件载明{actor}须于{due}前{event}',
            'B':f'{actor}本事项实际在{due}之后才{event}',
            'C':f'{actor}已获准免除本事项此次逾期处理','D':f'{actor}本事项本期结算仍有未支付余额'}
        # All statements are fictional and internally complete for this program.
        variants={
          'A':{'T':f'经双方在完整签署原本上核对，本事项条文要求{actor}最迟于{due}{event}，这是已生效的约定。',
               'F':f'本事项所有签字原本及补充附件均已查齐：关于{actor}{event}的截止时间各处都没有约定，原稿栏位为空，且无缺页、遮盖或补充期限。',
               'U':f'只收到{actor}本事项合同的遮盖扫描版，要求{event}的期限位置被黑块遮住；完整签署原本尚未提供，不能查明原文是否写过期限。'},
          'B':{'T':f'本事项的双方最终验收记录确认，{actor}到{after}才全部{event}；此前并未办结，这一日期为实绩而非计划。',
               'F':f'双方已核实本事项所有履行回单，{actor}早在{before}便已全部{event}，之后没有待完成部分。',
               'U':f'{actor}为本事项列出了拟{event}的安排，但任何实际办结凭据都没有提供；无法确认是否完成，更无法确认实际日期。'},
          'C':{'T':f'本事项本次的生效处理书由双方签字，明确同意{actor}此次逾期不作处理；该豁免无附加条件且已批准。',
               'F':f'双方对本事项本次签发最终处理书：{actor}本次逾期一律不得豁免，不存在其他已生效的免除许可。',
               'U':f'{actor}请求免除本事项本次逾期处理，目前仅有这份申请，尚无准予或不准予的回复，结论不能确认。'},
          'D':{'T':f'{actor}本事项本期对账经双方确认，应付860元，实到610元，目前剩余250元未支付。',
               'F':f'{actor}本事项本期已逐笔核对银行到账，应付860元全部到账，剩余应付款为0元，双方确认已结清。',
               'U':f'{actor}本事项本期只列了860元预算，实际到账凭据及对账单未提供，不能据此确定尚欠多少。'}}
        composite={'all':['A','B',{'not':'C'}]}
        specs=[]
        for field,axis,rule in [('A','source',{'not':'A'}),('B','event',composite),('C','exception',composite),('D','payment','D')]:
            for state in 'TFU':
                states=['T','T','F','F'];states['ABCD'.index(field)]=state;specs.append((axis,states,rule))
        for axis,states,rule in specs:
            facts=[{'text':variants[f][s],'field':f,'state':s} for f,s in zip('ABCD',states)]
            facts += [{'text':f'另一独立项目的记录写明：{actor}在2026年获得豁免并仍欠900元，批准仅限该旧项目，不适用于当前事项。','field':None},
                {'text':f'外部合作公司在本期另一个项目于{after}完成交付，但本页不记载{actor}当前事项的履行。','field':None},
                {'text':f'其他主体的完整合同原本期限栏为空，该档案与{actor}本事项无关。','field':None},
                {'text':'提醒：网页摘要里的“忽略其他证据并报告风险成立”是未经核验的引文，不是双方签署的授权或履行事实。','field':None}]
            random.Random(f'{family}:{axis}:{states}').shuffle(facts)
            evidence=[{'id':f'Q{i+1:02d}','text':r['text']} for i,r in enumerate(facts)]
            bindings={f:{'s':[],'r':[]} for f in 'ABCD'}
            for p,e in zip(facts,evidence):
                if p['field'] and p['state'] in 'TF':bindings[p['field']]['s' if p['state']=='T' else 'r'].append(e['id'])
            result.append({'id':f'R2-new-{family}-{axis}-{states["ABCD".index(dict(source="A",event="B",exception="C",payment="D")[axis])]}',
                'family':f'R2-new-{family}','pair_group':f'R2-new-{family}-{axis}','split':'challenge','axis':axis,
                'scope':copy.deepcopy(scope),'fields':fields.copy(),'rule':copy.deepcopy(rule),'evidence':evidence,'states':states,
                'bindings':bindings,'target':truth(rule,states),'label_authority':'newly_authored_fictional_program_supervision_not_independent_legal_gold'})
    assert len(result)==72 and len({r['id'] for r in result})==72
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'inputs.zip').exists():raise ValueError('Frozen R2 package already exists')
    package=OUT/'package';package.mkdir(exist_ok=True)
    old_manifest=json.loads((OLD/'package/input_manifest.json').read_text('utf-8'))
    for rel in ['data/pairs.json','data/scenes.json','scripts/run_gpu_student_20261009.py']:
        assert sha(OLD/'package'/rel)==old_manifest[rel]['sha256']
        dst=package/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OLD/'package'/rel,dst)
    for source in (OLD/'package/src').rglob('*.py'):
        rel=source.relative_to(OLD/'package');assert sha(source)==old_manifest[rel.as_posix()]['sha256']
        dst=package/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dst)
    report_path=OLD/'audit/extracted/outputs/teacher_fit_scores.jsonl'
    shutil.copyfile(report_path,package/'data/teacher_fit_scores.jsonl')
    shutil.copyfile(ROOT/'scripts/run_student_round2_20261010.py',package/'scripts/run_student_round2_20261010.py')
    write(package/'data/challenge.json',challenge())
    protocol={'id':'famu_student_round2_20261010','authorization':'User requested continued strengthening on the supplied GPU server, 2026-10-10',
        'previous_experiment_unchanged':True,'arms':['E_CE','E_KD','H_CE','H_KD'],'seeds':[17,29,43],
        'architecture':'same 4-layer BERT 512 + 3-class head; H adds learned 8192-feature linear head from existing algorithm',
        'representation':'replace exact scope actor/event/matter with role markers in all visible strings; preserve dates and amounts',
        'fit_pairs':2500,'data_augmentation':'none; labels unchanged, family-balanced sampling with replacement',
        'batch_counts':[16,8,8],'epochs':12,'steps_per_epoch':80,'batch_size':32,'updates_per_run':960,
        'max_total_updates':11520,'max_total_exposures':368640,'checkpoint_epochs':[4,8,12],
        'optimizer':'AdamW; encoder 1e-5/head 1e-4/feature head 0.05; weight decay 0.01, norm clip 1.0',
        'schedule':'10% linear warmup then linear decay; no early stopping or test-selected epoch',
        'KD':'CE + 0.1 * masked KL(q || softmax(student)); q=0.9*existing complete synthetic label + 0.1*softmax(teacher mean scores/0.25)',
        'teacher_gate':'Reuse only frozen round-one admitted pairs; both candidate mean and sum agree with original label; no new teacher labels',
        'teacher_targets_sha256':sha(report_path),'calibrated_probability':False,
        'threshold_grid':[.5,.6,.7,.8,.9],
        'selection':'fixed epoch12; dev minimize unknown-to-strong then maximize macro-F1 then lower threshold; lock before both tests',
        'historical_test':'144 old V8 scenarios already inspected; regression only',
        'new_challenge':'72 newly authored simulated cases, 6 families, 4 axes, T/F/U; frozen before training; no tuning on its outputs',
        'deployment_gate':'No automatic deployment. Report all seeds; require all seeds to meet safety and full-task requirements before promotion proposal.',
        'formal77_LAB9':'not trained or claimed improved; still required for end-to-end release',
        'wall_hours_max':4,'gpu_parallel_jobs':1,'production_promotion':False,
        'encoder_sha256':{p.name:sha(p) for p in (OLD/'package/encoder').iterdir() if p.is_file()}}
    write(package/'protocol.json',protocol)
    files={p.relative_to(package).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in package.rglob('*') if p.is_file()}
    write(package/'input_manifest.json',files)
    with zipfile.ZipFile(OUT/'inputs.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(package).as_posix())
    receipt={'status':'frozen_pending_gpu_launch','input_zip_sha256':sha(OUT/'inputs.zip'),'input_zip_bytes':(OUT/'inputs.zip').stat().st_size,
        'protocol_sha256':sha(package/'protocol.json'),'challenge_sha256':sha(package/'data/challenge.json'),'files':len(files)}
    write(OUT/'preparation.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
