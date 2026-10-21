"""R10 fictional language data with authored current-clause character spans."""
import copy
from scripts import student_round9_data as old
from scripts.prepare_student_round8_20261010 import SHORT

# These are authored synthetic paraphrases, not expert legal annotations.
CORE = {
'A': {'T':['双方签本连同增页均已查证；{a}{e}的最后日期写为{d}，该要求仍然有效。','将所有有效签页逐一核实后可以确定：{a}{e}不得迟过{d}，双方对此已签认。'],
      'F':['双方签本连同增页均已查证；{a}{e}没有被约定任何日期，并非文书未收全。','将所有有效签页逐一核实后可以确定：没有给{a}{e}限定时点，完整文书不存在这类条文。'],
      'U':['增页及部分签本无法调到；现在不能断定{a}{e}究竟约没约日期。','关于{a}{e}的有效签页没有凑齐；时限曾否约定仍处于不能验证的状态。']},
'B': {'T':['已验真的办结证明写明，{a}{e}于{late}全部完成，此前一直未全部完成。','{a}{e}的实际完结日经完整回单相互印证为{late}，早些时候仍有未办工作。'],
      'F':['已验真的办结证明写明，{a}{e}于{early}全部完成，其后没有未完成部分。','{a}{e}的实际完结日经完整回单相互印证为{early}，自该日起已无剩余工作。'],
      'U':['{late}是{a}{e}预排的日子而非验真办结证明；没有实际回单，结果无法判断。','{a}{e}只有{late}的意向安排，能够证明真实完成时间的单据至今未获取。']},
'C': {'T':['对方已正式签准{a}本次逾期的免责，决定已经发生效力，所附前提无一未满足。','{a}这次晚交获得了已生效的免处理答复；答复由有权人签署，所要求的条件均已兑现。'],
      'F':['对方已正式签拒{a}本次逾期的免责，拒绝已经发生效力；查全材料也没有另外的同意。','{a}这次晚交获得了已生效的不予免处理答复；完整有效文件中没有任何相反批准。'],
      'U':['对方只拟同意{a}本次逾期免责，但还要求补正；补正与批准生效均未证实，不能作为最终准许。','{a}这次晚交的免处理答复尚待条件兑现；兑现情况与最终签发都不明，有效同意或拒绝均无定论。']},
'D': {'T':['原账及真实到账均已逐项验真：{a}本期到期应交{total}元，仅入账{paid}元，{owed}元仍待付。','汇齐全部到期账目与收款原证后，{a}本期需付{total}元，已付{paid}元，尚差{owed}元。'],
      'F':['原账及真实到账均已逐项验真：{a}本期到期应交{total}元，已入账{total}元，未付0元且无其它到期账。','汇齐全部到期账目与收款原证后，{a}本期需付{total}元，已付{total}元，尚差0元，无其它到期款。'],
      'U':['{a}本期表上的{total}元仅为预计；原账及真实到账均未查实，不能计算是否还欠。','{a}本期虽有{total}元的粗算，但到期原账和真实收款都缺证，余额究竟多少无从核定。']}}
QUESTIONS = {
'A':['查齐签署凭据后能否确认{a}须赶在{d}前{e}','完整有效租约中，{a}{e}是否被约束不得晚于{d}'],
'B':['{a}{e}的真实完结时间，是否落在{d}以后','依据实际办结的原证，{a}{e}是否超过了{d}'],
'C':['{a}此次逾期免处理，是否已有前提满足且生效的同意','{a}这回迟延免责是否已获有权方有效准许'],
'D':['全部到期款核真之后，{a}本期是不是仍欠钱','核全应付与实收后，{a}本期到期租金是否尚有差额未交']}

def vals(split,family):
    prefix={'train':'榆汐','dev':'荷垣','test':'湛亭'}[split]
    a=f'{prefix}{family+1}承租事务社';m=f'{prefix}{family+1}住房承租核查事项'
    year=2051+family//9;month=family%9+2;total=3200+family*197;owed=210+family*17
    return dict(a=a,m=m,e=['退房移交钥匙','返还租住住房','办妥租房交接','退租交还房屋'][family%4],
                d=f'{year}年{month}月19日',early=f'{year}年{month}月11日',late=f'{year}年{month}月26日',
                total=total,paid=total-owed,owed=owed,period=f'{year}年{month}月')

def fact(f,s,style,v,compact=False):
    core=(SHORT[f][s][style%3] if style<4 else '{m}'+CORE[f][s][style-4]).format(**v)
    if compact:return core
    other='F' if s=='T' else 'T'
    quote=(SHORT[f][other][(style+1)%3] if style<4 else '{m}'+CORE[f][other][5-style]).format(**v)
    templates=[
        ('抄录人员留下了失效说法“'+quote+'”，这段话仅供追溯并已撤销。正文实际采用的内容是：',core,'核验结论不受失效抄录影响。'),
        ('先列这次已核验的正文：',core,'附件另存“'+quote+'”，但是该附件未曾生效，也不是对正文的更正。'),
        ('草案所列“'+quote+'”没有获签，不能据此判断。签署后的核验记录写的是：',core,'未生效草案没有替代以上记录。'),
        ('现行核验资料载明：',core,'记录员也提到过“'+quote+'”；经复查该旧记已经作废，仅留作核错线索。'),
        ('档案附注保留“'+quote+'”，但明示不予采信。有效核验段落另记：',core,'附注既不变更本段，也不构成相反的有效材料。'),
        ('一度传抄的“'+quote+'”已经撤销，存档只为追溯讹误，不具事实证明用途。本轮核验所能确定的情况则是：',core,'')]
    return ''.join(templates[style])

def generate():
    # Reuse only the scene/rule combinator, never an old held-out wording bank.
    saved=(old.vals,old.fact,old.QNEW)
    try:
        old.vals,old.fact,old.QNEW=vals,fact,QUESTIONS
        rows,pairs=old.generate()
    finally:old.vals,old.fact,old.QNEW=saved
    for row in rows:
        for k in ['id','family','pair_group']:row[k]=row[k].replace('R9-','R10-')
        split=row['split'].split('_')[0];family=int(row['family'].rsplit('-',1)[1]);v=vals(split,family)
        gold={e['id']:[] for e in row['evidence']}
        for f,s in zip('ABCD',row['states']):
            core=fact(f,s,row['style_id'],v,True);found=[]
            for e in row['evidence']:
                start=e['text'].find(core)
                if start>=0:
                    assert e['text'].find(core,start+1)<0
                    found.append(e['id']);gold[e['id']].append({'start':start,'end':start+len(core),
                                                            'class':2 if s=='U' else 1,'field':f})
            assert len(found)==1,(row['id'],f,found)
        row['span_gold']=gold
    for p in pairs:p['group']=p['group'].replace('R9-','R10-')
    paragraphs={}
    for row in rows:
        if row['split']!='train':continue
        for e in row['evidence']:
            item={'text':e['text'],'scope':copy.deepcopy(row['scope']),'spans':row['span_gold'][e['id']],
                  'group':row['family']}
            from src.evidence_reasoning.local_field_student import digest
            key=digest({'text':e['text'],'scope':row['scope']})
            if key in paragraphs:assert paragraphs[key]==item
            paragraphs[key]=item
    for item in list(paragraphs.values()):
        if not item['spans']:continue
        assert len(item['spans'])==1
        span=item['spans'][0];text=item['text'][span['start']:span['end']]
        compact=item|{'text':text,'spans':[span|{'start':0,'end':len(text)}]}
        key=digest({'text':text,'scope':item['scope']})
        if key in paragraphs:assert paragraphs[key]==compact
        paragraphs[key]=compact
    return rows,pairs,list(paragraphs.values())
