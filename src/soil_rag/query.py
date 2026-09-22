"""Intent classification, slot extraction, query rewriting and expansion."""

from __future__ import annotations

import re
from collections import OrderedDict

from .schemas import Intent, QueryPlan


INTENT_KEYWORDS: OrderedDict[Intent, tuple[str, ...]] = OrderedDict(
    [
        (Intent.POLICY_QUERY, ("政策", "规定", "条款", "依据", "文件")),
        (Intent.PROCESS_CONSULTATION, ("流程", "怎么办", "办理", "审批", "审查")),
        (Intent.MATERIAL_LIST, ("材料", "资料", "清单", "提交", "申报")),
        (Intent.PROJECT_SCOPE, ("范围", "责任范围", "扰动范围", "占地")),
        (Intent.TECHNICAL_MEASURE, ("措施", "排水", "拦挡", "苫盖", "植被", "表土")),
        (Intent.MONITORING_EXPLANATION, ("监测", "监测报告", "监测点", "水土流失监测")),
        (Intent.ACCEPTANCE_CONSULTATION, ("验收", "自主验收", "验收报告", "核查")),
        (Intent.FEE_STANDARD, ("费用", "补偿费", "收费", "缴费", "标准")),
        (Intent.TIME_LIMIT, ("多久", "时限", "期限", "多长时间", "时间")),
        (Intent.REGION_REQUIREMENT, ("本地", "当地", "地区", "省", "市", "自治区")),
        (Intent.TERM_EXPLANATION, ("什么是", "解释", "含义", "定义", "概念")),
    ]
)


class RuleBasedIntentClassifier:
    def classify(self, query: str) -> Intent:
        scores = {intent: sum(query.count(keyword) for keyword in keywords) for intent, keywords in INTENT_KEYWORDS.items()}
        best_intent, best_score = max(scores.items(), key=lambda item: item[1])
        return best_intent if best_score else Intent.OTHER


class SlotExtractor:
    region_re = re.compile(r"(北京|天津|上海|重庆|河北|山西|辽宁|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|广西|海南|四川|贵州|云南|陕西|甘肃|青海|内蒙古|西藏|宁夏|新疆)(?:省|市|自治区|特别行政区)?")
    stage_re = re.compile(r"(立项|设计|施工|竣工|验收|报批|审查|监测|监理)阶段?")
    project_re = re.compile(r"([\u4e00-\u9fff]{0,8}(?:项目|工程))")

    def extract(self, query: str) -> dict[str, str]:
        slots: dict[str, str] = {}
        if match := self.region_re.search(query):
            slots["region"] = match.group(0)
        if match := self.stage_re.search(query):
            slots["stage"] = match.group(1)
        if match := self.project_re.search(query):
            slots["project_type"] = match.group(1)
        slots["current"] = "true" if any(word in query for word in ("现在", "当前", "有效", "最新")) else "false"
        return slots


class QueryRewriter:
    replacements = {
        "要不要批": "是否需要办理水土保持审批",
        "要不要报": "是否需要报批水土保持方案",
        "要准备啥": "需要准备哪些申报材料",
        "多久能办完": "办理时限和审批周期是多少",
        "怎么弄": "办理流程和前置条件是什么",
        "堆土": "弃土弃渣和临时堆土",
    }

    def rewrite(self, query: str, slots: dict[str, str]) -> str:
        rewritten = query.strip()
        for source, target in self.replacements.items():
            rewritten = rewritten.replace(source, target)
        qualifiers = []
        for key in ("project_type", "region", "stage"):
            if slots.get(key):
                qualifiers.append(f"{key}={slots[key]}")
        if slots.get("current") == "true":
            qualifiers.append("时效=当前有效版本")
        return f"{rewritten}（{'；'.join(qualifiers)}）" if qualifiers else rewritten


class QueryExpander:
    synonyms = {
        "水土保持方案": ("水保方案", "水土保持报告", "水土保持方案报告书"),
        "审批": ("报批", "审查", "审批手续"),
        "材料": ("资料", "申报材料", "附件"),
        "弃土弃渣": ("弃土场", "弃渣场", "临时堆土"),
        "监测": ("水土保持监测", "监测报告", "监测点位"),
    }

    def expand(self, query: str, intent: Intent) -> tuple[str, ...]:
        expansions = [query]
        for term, synonyms in self.synonyms.items():
            if term in query:
                expansions.extend(query.replace(term, synonym) for synonym in synonyms)
        if intent == Intent.MATERIAL_LIST:
            expansions.append(f"{query} 基础资料 申报清单 缺失材料")
        elif intent == Intent.PROCESS_CONSULTATION:
            expansions.append(f"{query} 前置条件 办理节点 审查流程")
        elif intent == Intent.TERM_EXPLANATION:
            expansions.append(f"{query} 定义 适用条件 技术含义")
        return tuple(dict.fromkeys(expansions))


class QueryPlanner:
    def __init__(self):
        self.classifier = RuleBasedIntentClassifier()
        self.slots = SlotExtractor()
        self.rewriter = QueryRewriter()
        self.expander = QueryExpander()

    def plan(self, query: str) -> QueryPlan:
        intent = self.classifier.classify(query)
        slots = self.slots.extract(query)
        rewritten = self.rewriter.rewrite(query, slots)
        expansions = self.expander.expand(rewritten, intent)
        return QueryPlan(
            original_query=query,
            intent=intent,
            rewritten_query=rewritten,
            expansions=expansions,
            slots=slots,
            require_current=slots.get("current") == "true",
            use_graph=intent not in {Intent.OTHER, Intent.TERM_EXPLANATION},
        )
