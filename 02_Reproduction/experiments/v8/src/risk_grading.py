"""Team-owned severity semantics; never derive severity from proposition truth."""
VERSION = "2026.09.29-four-level-v1"
LEVELS = ("low", "medium", "high", "critical")
DEFINITIONS = {
    "critical": "会立即导致重大资金、就业、居住、法律责任风险。",
    "high": "明显失衡且一旦触发损失较大。",
    "medium": "存在明显模糊或偏向，但可通过补充条款修正。",
    "low": "提示性优化项。",
}
NAMES = dict(zip(LEVELS, ("提示风险（低风险）", "中风险", "高风险", "重大风险")))
PROPOSITION_TYPES = {"contract_content", "normative_sufficiency", "case_application"}
RISK_STATES = {"established", "not_established", "unknown", "unlabeled", "technical_failure"}


def normalize_grade(value):
    if value is None or value == "":
        return None
    value = str(value).strip().lower()
    aliases = {**{k: k for k in LEVELS}, **{v: k for k, v in NAMES.items()},
               "低": "low", "低风险": "low", "提示风险": "low", "中": "medium", "中等": "medium",
               "高": "high", "重大": "critical"}
    if not value:
        return None
    if value not in aliases:
        raise ValueError("Unknown severity; missing labels must not be filled")
    return aliases[value]


def validate_severity_basis(level, basis):
    """Validate completeness, not legal truth or the substantive grade itself."""
    level = normalize_grade(level)
    if level is None:
        return False
    required = ("risk_object", "trigger", "consequence", "magnitude", "remediability")
    if not isinstance(basis, dict) or any(not isinstance(basis.get(k), str) or not basis[k].strip() for k in required):
        raise ValueError("Severity needs object, trigger, consequence, magnitude and remediability")
    if not isinstance(basis.get("unconfirmed_facts"), list):
        raise ValueError("Record unconfirmed facts explicitly")
    if level == "critical" and (basis.get("immediate") is not True or not basis.get("immediacy_basis")
                                or basis.get("risk_domain") not in {"funds", "employment", "housing", "legal_liability"}):
        raise ValueError("Critical requires immediate major harm domain and its explicit basis")
    return True


def risk_result(status="unlabeled", severity=None, basis=None):
    if status not in RISK_STATES:
        raise ValueError("Risk status must distinguish unknown, unlabeled and technical failure")
    level = normalize_grade(severity)
    if level is not None:
        if status != "established":
            raise ValueError("Severity requires an explicitly established risk object")
        validate_severity_basis(level, basis)
    return {"risk_established_status": status,
            "risk_established": True if status == "established" else False if status == "not_established" else None,
            "severity": level, "severity_basis": basis, "severity_standard_version": VERSION,
            "severity_loss_mask": int(level is not None)}


def prompt_standard():
    return "严重度规则：" + "；".join(f"{k}={v.rstrip('。')}" for k, v in DEFINITIONS.items()) + "。" + (
        "命题T/F/U、风险是否成立、严重度和工作流状态分别判断；U不等于低风险，F不等于无风险。"
        "重大风险必须说明风险对象、触发前提、立即性与重大后果依据；依据缺失时保留待复核，不能自动降为低风险。"
        "充分性问题与实际履行事实分题；不把未证明例外成立当成例外不成立。")
