"""Authored R11 synthetic current-source spans; no old test material used for fitting."""
from scripts import student_round10_data as common
CORE={
'A':{'T':['租房合同已齐备并核真，{a}{e}的约定截止日是{d}。','核对承租协议的全部有效页后，{a}必须在{d}以内{e}。','双方生效文本逐页对过，{a}{e}的最后期限确为{d}。'],
     'F':['租房合同已齐备并核真，对{a}{e}从未约定截止日。','核对承租协议的全部有效页后，确定没有条款限制{a}{e}的日期。','双方生效文本逐页对过，未给{a}{e}设定任何最后期限。'],
     'U':['租房合同缺少补页，{a}{e}到底有无截止日尚不能核实。','承租协议的有效页并未集齐，{a}{e}是否受时限约束没有足够凭据。','双方生效文本有关键页缺失，无法查明{a}{e}是否约过最后期限。']},
'B':{'T':['{a}{e}实际全部办完是在{late}，该日以前未办完，验真原件可证。','原始交接凭条经核对，{a}{e}完成于{late}，不是在此前完成。','全部验收原证相互吻合，{a}{e}直至{late}方全部办妥。'],
     'F':['{a}{e}实际全部办完是在{early}，自该日起没有遗留项目，验真原件可证。','原始交接凭条经核对，{a}{e}完成于{early}，此后已无未完成项目。','全部验收原证相互吻合，{a}{e}早在{early}即全部办妥。'],
     'U':['{a}{e}所记{late}只是计划，实际交接原件没有找到，完成与否不清楚。','{a}{e}拟于{late}完成；尚无原始交接凭条，真实完成时点无法查定。','{a}{e}把{late}列作预定日，实际验收原证缺失，不能当成已办妥。']},
'C':{'T':['{a}本次晚交的免责已经有权方批准生效，所有附加条件均已满足。','有权人针对{a}这次迟延签发了有效免责决定，要求的前提已全部落实。','针对{a}这一回超期，有权方已作出发生效力的豁免，生效条件全部成就。'],
     'F':['{a}本次晚交的免责已经有权方明确拒绝，资料收齐也找不到有效批准。','有权人针对{a}这次迟延签发了有效拒绝决定，全部材料均无另行免责。','针对{a}这一回超期，有权方已作出发生效力的不予豁免，完整记录不存在另批。'],
     'U':['{a}本次晚交的免责只在草拟中，附加条件是否满足和批准是否生效均未查明。','{a}这次迟延的免责仍须满足前提，前提及最终签发情况都缺证，尚无结论。','针对{a}这一回超期只有待核准的豁免意向，条件成就与否和正式效力均不明。']},
'D':{'T':['{a}本期应付{total}元、实付{paid}元、仍欠{owed}元，全部到期账和收款已核真。','所有到期租金与收据对清后，{a}应交{total}元，到账{paid}元，未交{owed}元。','到期总账和实收凭证一一核实，{a}应付合计{total}元，实收{paid}元，余额{owed}元未付。'],
     'F':['{a}本期应付{total}元、实付{total}元、仍欠0元，全部到期账和收款已核真且无其它欠款。','所有到期租金与收据对清后，{a}应交{total}元，到账{total}元，未交0元，无其它到期账。','到期总账和实收凭证一一核实，{a}应付合计{total}元，实收{total}元，余额0元且没有别的到期款。'],
     'U':['{a}本期{total}元只是估数，到期账和收款原证缺失，是否欠款无法确定。','{a}租金暂算{total}元，全部到期租金与收据尚未核真，不能查定未付余额。','{a}表上{total}元未获原始到期总账及实收凭证支持，欠款余额仍无法核实。']}}
QUESTIONS={
'A':['租房约定核齐后，能否确定{a}最迟应在{d}{e}','有效承租文本是否要求{a}{e}不能超过{d}'],
'B':['实际核真的交接结果是否表明{a}{e}晚于{d}','有无原证确认{a}{e}是在{d}之后才办妥'],
'C':['{a}此次晚交是否已取得所有条件成就的有效免责','有权人是否已有效豁免{a}本次超期且前提全部满足'],
'D':['{a}本期已到期的房租核清实收后是否还有钱未交','核齐到期总账和到账凭据，{a}本期是否还存在租金缺口']}

def vals(split,family):
    p={'train':'松澄','dev':'芦堤','test':'砚湾'}[split];year=2058+family//8;month=family%8+2;total=4100+family*211;owed=280+family*19
    return dict(a=f'{p}{family+1}租住服务社',m=f'{p}{family+1}住宅承租办理事项',e=['归还租房钥匙','完成住房交接','退租返还住房','交还承租房屋'][family%4],d=f'{year}年{month}月17日',early=f'{year}年{month}月9日',late=f'{year}年{month}月24日',total=total,paid=total-owed,owed=owed,period=f'{year}年{month}月')

def fact(f,s,style,v,compact=False):
    bank=0 if style<4 else style-3
    core=('{m}：'+CORE[f][s][bank]).format(**v)
    if compact:return core
    quote=('{m}：'+CORE[f]['F' if s=='T' else 'T'][bank]).format(**v)
    wraps=[('旧口述曾称“'+quote+'”。该口述已撤回；经原件查证，目前有效内容为：',core,'请勿将撤回内容混入核验事实。'),
      ('经查证现行内容为：',core,'末尾附存“'+quote+'”，该项仅是未获确认的旧笔记，已经作废。'),
      ('登记底稿所记“'+quote+'”未获采纳；纠正并核实后，事实段应读作：',core,'底稿只用来追溯错误。'),
      ('不受过期抄本影响的有效记载如下：',core,'另有抄本“'+quote+'”，现已失效且不得作为相反事实。'),
      ('复核者说明：虽然附件曾摘录“'+quote+'”，但该摘录已经被撤销；其后查证的现行内容是：',core,'先前摘录没有证明作用。'),
      ('追溯材料中保留“'+quote+'”；那只是失效传抄，不作事实使用。本轮核验所能确定的情况则是：',core,'')]
    return ''.join(wraps[style])

def generate():
    saved=(common.vals,common.fact,common.QUESTIONS)
    try:
        common.vals,common.fact,common.QUESTIONS=vals,fact,QUESTIONS
        rows,pairs,paragraphs=common.generate()
    finally:common.vals,common.fact,common.QUESTIONS=saved
    for row in rows:
        for k in ['id','family','pair_group']:row[k]=row[k].replace('R10-','R11-')
    for p in pairs+paragraphs:p['group']=p['group'].replace('R10-','R11-')
    return rows,pairs,paragraphs

def replay(original):
    from src.evidence_reasoning.local_field_student import digest
    bank={}
    for p in original:
        if p['label'] not in [1,2]:continue
        x={k:p[k] for k in ['text','scope','group']};x['spans']=[{'start':0,'end':len(x['text']),'class':1}]
        key=digest({'text':x['text'],'scope':x['scope']})
        if key in bank:assert bank[key]==x
        bank[key]=x
    assert bank
    return list(bank.values())
