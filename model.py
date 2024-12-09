import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LayerNormalization, Dropout, Dense, Input, MultiHeadAttention, Layer, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba
import json
import random
import re
import time
import pickle
import os
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from colorama import Fore, Style, init

# 初始化colorama，自动重置颜色
init(autoreset=True)

# 定义一些常用词语的同义词映射
synonyms_dict = {
    "你好": ["您好", "嗨", "早安", "你好啊", "问好"],
    "谢谢": ["多谢", "感谢", "感激", "谢谢你", "多谢你"],
    "再见": ["拜拜", "再会", "回见", "再见了", "告别"],
        "今天": ["今日", "今天", "今天早上", "这天", "今日一日"],
    "吃": ["进食", "用餐", "吃饭", "进餐", "用膳"],
    "喜欢": ["偏爱", "喜好", "爱好", "钟情", "热爱"],
    "学习": ["研读", "自学", "学问", "进修", "求知"],
    "工作": ["任务", "职业", "工作", "职务", "岗位"],
    "地方": ["位置", "地区", "地点", "场所", "区域"],
    "时间": ["时光", "时段", "时间段", "时刻", "时机"],
    "快": ["迅速", "快速", "迅捷", "飞快", "快捷"],
    "慢": ["迟缓", "缓慢", "徐缓", "慢悠悠", "拖沓"],
    "漂亮": ["美丽", "好看", "迷人", "秀丽", "亮丽"],
    "愉快": ["快乐", "开心", "幸福", "愉悦", "高兴"],
    "高兴": ["喜悦", "兴奋", "愉悦", "激动", "喜乐"],
    "家": ["住所", "家庭", "家园", "家里", "住宅"],
    "爱": ["喜爱", "钟爱", "喜好", "热爱", "倾心"],
    "手机": ["电话", "智能机", "移动电话", "智能手机", "手机设备"],
    "汽车": ["车辆", "小车", "轿车", "车子", "交通工具"],
    "电影": ["影片", "影像", "影片", "电影片", "影视"],
    "音乐": ["歌曲", "旋律", "乐曲", "音符", "乐音"],
    "书": ["图书", "书籍", "刊物", "书本", "资料"],
    "高": ["高度", "高高", "巨高", "高耸", "崇高"],
    "低": ["矮", "低下", "底", "低矮", "浅"],
    "明天": ["明日", "第二天", "未来的日子", "来日", "明天早上"],
    "昨晚": ["昨夜", "前晚", "昨日晚上", "昨晚的", "前一晚"],
    "爱好": ["兴趣", "偏好", "爱好", "喜好", "癖好"],
    "好": ["优秀", "良好", "美好", "杰出", "卓越"],
    "差": ["差劲", "不行", "不佳", "差得远", "劣"],
    "美丽": ["美丽", "动人", "迷人", "迷人", "秀美"],
    "糟糕": ["糟糕", "差劲", "惨不忍睹", "不堪", "烂透了"],
    "高兴": ["开心", "兴奋", "愉快", "欣喜", "满足"],
    "忧虑": ["担忧", "焦虑", "担心", "忧心", "不安"],
    "生气": ["愤怒", "不悦", "气愤", "愤慨", "愠怒"],
    "冷": ["寒冷", "冰冷", "凉爽", "寒气", "刺骨"],
    "热": ["炎热", "高温", "炙热", "炽热", "燥热"],
    "快速": ["迅速", "飞快", "快捷", "急速", "高速"],
    "精力": ["活力", "能量", "活跃", "精神", "动力"],
    "安静": ["安宁", "寂静", "静默", "清幽", "安祥"],
    "嘈杂": ["喧嚣", "吵闹", "杂乱", "嘈杂", "喧闹"],
    "方便": ["便捷", "简便", "轻松", "容易", "省事"],
    "复杂": ["难", "复杂", "麻烦", "繁琐", "繁杂"],
    "简单": ["简单", "容易", "简易", "直接", "朴素"],
    "容易": ["轻松", "简单", "容易", "容易理解", "手到擒来"],
    "难": ["困难", "艰难", "不易", "困难重重", "困难险阻"],
    "有趣": ["有意思", "好玩", "有趣", "有吸引力", "风趣"],
    "无聊": ["枯燥", "无趣", "乏味", "单调", "沉闷"],
    "开心": ["快乐", "高兴", "愉快", "兴奋", "欣喜"],
    "恐惧": ["害怕", "恐慌", "惊慌", "惶恐", "惧怕"],
    "信任": ["相信", "依赖", "信赖", "相信", "托付"],
    "害怕": ["恐惧", "害怕", "畏惧", "恐怖", "惧怕"],
    "幸运": ["好运", "幸运", "天赐", "幸运的", "顺利"],
    "失败": ["失利", "倒闭", "败北", "失误", "破产"],
    "成功": ["胜利", "成功", "达成", "成就", "凯旋"],
    "赚钱": ["盈利", "获利", "赚钱", "挣钱", "盈利收入"],
    "花钱": ["消费", "支出", "花费", "浪费", "支付"],
    "努力": ["奋斗", "努力", "拼搏", "尽力", "奋发"],
    "懒惰": ["懒散", "不勤快", "不努力", "懒惰", "无所事事"],
    "聪明": ["机智", "聪慧", "智慧", "聪颖", "才智"],
    "笨": ["迟钝", "愚笨", "不聪明", "傻", "笨拙"],
    "合作": ["协作", "共同", "配合", "协同", "合力"],
    "竞争": ["对抗", "竞争", "较量", "竞赛", "争夺"],
    "变化": ["改变", "调整", "变化", "变动", "革新"],
    "稳定": ["平稳", "稳固", "稳定", "牢固", "坚固"],
    "勇气": ["胆量", "勇敢", "胆识", "勇气", "豪气"],
    "智慧": ["聪明", "睿智", "智慧", "才智", "机敏"],
    "善良": ["仁慈", "良善", "心地善良", "慈悲", "温良"],
    "自由": ["自主", "自由", "独立", "自发", "解放"],
    "贫穷": ["贫困", "拮据", "贫寒", "落魄", "穷困"],
    "富有": ["富裕", "富饶", "殷实", "有钱", "富贵"],
    "勇敢": ["勇气", "大胆", "勇敢", "果敢", "英勇"],
    "创造": ["创新", "创造", "开创", "发明", "制作"],
    "坚持": ["维持", "坚持", "保持", "持续", "固守"],
    "疑惑": ["疑问", "不解", "疑虑", "困惑", "迷茫"],
    "希望": ["期望", "希望", "盼望", "期待", "渴望"],
    "哭": ["哭泣", "流泪", "悲伤", "号哭", "哭喊"],
    "笑": ["微笑", "笑容", "笑声", "哈哈", "开怀大笑"],
    "幸福": ["快乐", "幸运", "福气", "美满", "愉悦"],
    "寂寞": ["孤单", "寥落", "冷清", "孤独", "空虚"],
    "耐心": ["耐性", "坚韧", "持久", "耐力", "沉稳"],
    "痛苦": ["煎熬", "苦楚", "痛楚", "折磨", "悲伤"],
    "希望": ["期待", "愿望", "憧憬", "梦想", "期望"],
    "困扰": ["烦恼", "困惑", "麻烦", "障碍", "烦忧"],
    "责任": ["义务", "任务", "职责", "担子", "负担"],
    "自信": ["自信心", "信心", "自负", "信赖", "确定"],
    "聪慧": ["机智", "聪明", "睿智", "智慧", "聪颖"],
    "勇气": ["胆量", "勇敢", "胆识", "勇气十足", "英雄气概"],
    "烦恼": ["忧虑", "困扰", "烦忧", "焦虑", "忧愁"],
    "温暖": ["温馨", "暖和", "热情", "温情", "和煦"],
    "平静": ["安宁", "宁静", "沉静", "安稳", "静谧"],
    "无奈": ["无助", "无可奈何", "无计可施", "心有余而力不足", "无从下手"],
    "健康": ["身体好", "康健", "健壮", "强壮", "安康"],
    "孤独": ["寂寞", "孤单", "独自", "孤寂", "冷清"],
    "渴望": ["期待", "向往", "愿望", "盼望", "希冀"],
    "成熟": ["老练", "稳重", "深沉", "圆熟", "有经验"],
    "失败": ["失利", "败北", "错失", "落败", "败局"],
    "成功": ["胜利", "成就", "达成", "完美", "成功的"],
    "轻松": ["自在", "轻快", "愉快", "安逸", "舒适"],
    "压力": ["负担", "紧张", "困扰", "压力山大", "重负"],
    "梦想": ["理想", "憧憬", "追求", "愿望", "幻想"],
    "失望": ["沮丧", "丧气", "灰心", "落空", "泄气"],
    "幸福": ["快乐", "愉快", "美满", "幸运", "福气"],
    "懒惰": ["懒散", "不勤快", "懈怠", "懒惰", "偷懒"],
    "努力": ["奋斗", "拼搏", "勤奋", "用功", "努力向上"],
    "粗心": ["马虎", "不细心", "疏忽", "草率", "不小心"],
    "精致": ["精美", "细致", "优雅", "精巧", "美丽"],
    "悲伤": ["难过", "痛苦", "伤心", "沮丧", "忧伤"],
    "温柔": ["柔和", "温和", "亲切", "细腻", "娇柔"],
    "聪明": ["智慧", "机智", "明智", "聪慧", "才智"],
    "愚蠢": ["笨", "傻", "愚笨", "迟钝", "蠢"],
    "平凡": ["普通", "一般", "常见", "平淡", "无奇"],
    "独特": ["特别", "独一无二", "独特", "与众不同", "非凡"],
    "安静": ["宁静", "平静", "寂静", "安宁", "静谧"],
    "嘈杂": ["喧闹", "嘈杂", "吵闹", "杂乱", "喧哗"],
    "节省": ["节约", "节俭", "节省开支", "省钱", "紧缩"],
    "奢侈": ["豪华", "奢华", "奢侈浪费", "奢侈品", "铺张浪费"],
    "真诚": ["真挚", "诚恳", "真心", "真情", "诚实"],
    "虚伪": ["做作", "假装", "伪善", "不真诚", "矫情"],
    "善良": ["仁慈", "良善", "宽厚", "和蔼", "温和"],
    "恶劣": ["恶毒", "恶性", "险恶", "邪恶", "坏"],
    "热情": ["热烈", "热心", "积极", "充满激情", "真诚"],
    "冷淡": ["冷漠", "漠不关心", "疏远", "不关心", "淡漠"],
    "友好": ["友善", "和睦", "亲切", "温和", "热情"],
    "暴力": ["粗暴", "暴虐", "野蛮", "强暴", "粗鲁"],
    "冷静": ["镇定", "冷静", "理智", "沉着", "平静"],
    "冲动": ["激动", "冲动", "急躁", "暴躁", "鲁莽"],
    "创造": ["创新", "发明", "创作", "开创", "造就"],
    "模糊": ["不清晰", "模糊不清", "朦胧", "模糊不明", "含糊"],
    "清晰": ["明确", "清楚", "明了", "明晰", "透彻"],
    "强大": ["强劲", "强壮", "雄厚", "有力", "有力气"],
    "弱小": ["脆弱", "虚弱", "软弱", "孱弱", "不足"],
    "坚强": ["坚定", "坚韧", "强硬", "强大", "不屈不挠"],
    "脆弱": ["脆弱", "弱小", "不堪", "敏感", "易碎"],
    "包容": ["宽容", "容忍", "接纳", "原谅", "忍让"],
    "刻薄": ["尖刻", "刻毒", "冷酷", "苛刻", "刻意"],
    "关心": ["在意", "关爱", "照顾", "挂念", "体贴"],
    "忽视": ["忽略", "漠视", "不理", "冷落", "不顾"],
    "改变": ["转变", "变化", "调整", "改革", "变更"],
    "坚持": ["坚持不懈", "固守", "保持", "维持", "不放弃"],
    "放弃": ["放下", "抛弃", "舍弃", "遗弃", "停止"],
    "追求": ["追寻", "追赶", "寻找", "追逐", "渴望"],
    "迟到": ["耽误", "拖延", "延迟", "晚到", "滞后"],
    "准时": ["如约", "按时", "及时", "准点", "按时到达"],
    "温暖": ["温热", "温馨", "和煦", "热情", "融化"],
    "严肃": ["认真", "庄重", "严厉", "郑重", "威严"],
    "幽默": ["搞笑", "风趣", "滑稽", "诙谐", "逗趣"],
    "严谨": ["谨慎", "细致", "周密", "周到", "精确"],
    "粗心": ["马虎", "不细心", "疏忽", "草率", "不注意"],
    "聪明": ["机智", "智慧", "聪慧", "明智", "灵巧"],
    "愚蠢": ["笨", "傻", "迟钝", "愚笨", "愚昧"],
    "单纯": ["简单", "天真", "纯粹", "朴实", "直白"],
    "复杂": ["难", "复杂", "繁杂", "纷繁", "曲折"],
    "单调": ["枯燥", "乏味", "无趣", "重复", "平淡"],
    "新颖": ["独特", "创新", "新奇", "前卫", "别致"],
    "传统": ["古老", "经典", "习惯", "常规", "老旧"],
    "时尚": ["潮流", "流行", "时髦", "新潮", "现代"],
    "成熟": ["稳重", "老练", "深沉", "圆熟", "经验丰富"],
    "幼稚": ["天真", "稚嫩", "不成熟", "幼小", "不懂事"],
    "富裕": ["富有", "富足", "富裕", "殷实", "豪富"],
    "勇敢": ["无畏", "英勇", "大胆", "果敢", "胆大"],
    "智慧": ["聪慧", "睿智", "才智", "机智", "聪明"],
    "快乐": ["开心", "愉快", "幸福", "兴奋", "愉悦"],
    "悲伤": ["痛苦", "忧伤", "伤心", "难过", "沮丧"],
    "疲劳": ["劳累", "疲倦", "乏力", "困倦", "疲惫"],
    "安全": ["平安", "无害", "稳妥", "安稳", "无恙"],
    "风险": ["危险", "险情", "危机", "风险", "不确定性"],
    "爱情": ["恋爱", "爱意", "深情", "钟情", "倾心"],
    "友谊": ["友情", "友爱", "伙伴", "情谊", "友情关系"],
    "家庭": ["家", "家庭生活", "家园", "家族", "亲属"],
    "自由": ["解放", "自主", "自由意志", "独立", "开放"],
    "孤独": ["寂寞", "孤单", "孤立", "单独", "独自"],
    "荣耀": ["荣誉", "光荣", "威光", "辉煌", "显赫"],
    "羞愧": ["惭愧", "内疚", "羞耻", "惭愧", "愧疚"],
    "幽默": ["风趣", "搞笑", "滑稽", "诙谐", "幽默感"],
    "智慧": ["机智", "聪明", "聪慧", "睿智", "才智"],
    "冷静": ["镇定", "理智", "冷酷", "沉着", "从容"],
    "热情": ["激情", "活力", "热心", "友好", "热烈"],
    "自信": ["自尊", "信心", "自负", "信任", "自豪"],
    "放松": ["舒适", "轻松", "休息", "放缓", "松弛"],
    "困难": ["难题", "难关", "困境", "挑战", "艰难"],
    "乐观": ["积极", "阳光", "开朗", "豁达", "向上"],
    "悲观": ["消极", "灰心", "悲伤", "低落", "沮丧"],
    "懒惰": ["懒散", "不勤快", "懈怠", "懒散", "懒床"],
    "努力": ["拼搏", "奋斗", "努力工作", "尽力", "用功"],
    "激动": ["兴奋", "激烈", "紧张", "冲动", "亢奋"],
    "安静": ["宁静", "寂静", "平静", "安宁", "沉静"],
    "嘈杂": ["吵闹", "喧闹", "杂乱", "嘈杂", "喧嚣"],
    "独立": ["自主", "自立", "独立自主", "自给自足", "独立思考"],
    "依赖": ["依靠", "依附", "依赖性", "依赖于", "倚赖"],
    "温暖": ["温热", "热情", "和煦", "温馨", "热情洋溢"],
    "冷酷": ["严寒", "冷漠", "无情", "冷静", "冷淡"],
    "懒散": ["懒惰", "不勤快", "懒洋洋", "拖延", "不积极"],
    "勇气": ["胆量", "胆识", "勇敢", "决心", "豪气"],
    "宽容": ["包容", "容忍", "大度", "宽大", "宽心"],
    "狭隘": ["偏狭", "局限", "有限", "封闭", "狭小"],
    "充实": ["丰富", "饱满", "充分", "充裕", "完满"],
    "空虚": ["寂寞", "空洞", "虚无", "空旷", "无聊"],
    "光明": ["明亮", "光辉", "光亮", "璀璨", "亮丽"],
    "黑暗": ["阴暗", "昏暗", "黑沉沉", "漆黑", "无光"],
    "繁忙": ["忙碌", "繁杂", "忙碌不堪", "忙乱", "事务繁重"],
    "空闲": ["闲暇", "空暇", "空余", "闲散", "休闲"],
    "勇敢": ["大胆", "无畏", "英勇", "果敢", "刚强"],
    "内心": ["心灵", "心情", "内心世界", "内心深处", "情感"],
    "表面": ["外表", "外貌", "外观", "外形", "面容"],
    "高兴": ["开心", "愉快", "喜悦", "兴奋", "欣喜"],
    "伤心": ["痛心", "悲伤", "难过", "忧伤", "沮丧"],
    "沉默": ["静默", "无言", "默不作声", "寂静", "沉静"],
    "聒噪": ["嘈杂", "喧嚣", "吵闹", "嘈杂声", "杂乱无章"],
    "热爱": ["钟爱", "热衷", "喜爱", "倾心", "喜好"],
    "冷淡": ["冷漠", "疏远", "不关心", "无动于衷", "无兴趣"],
    "专注": ["集中", "聚焦", "全神贯注", "专心", "注重"],
    "分心": ["走神", "心不在焉", "分散注意力", "不专心", "心思不集中"],
    "简单": ["简易", "直白", "容易", "清晰", "朴素"],
    "复杂": ["难", "繁琐", "复杂多变", "混乱", "繁杂"],
    "安慰": ["慰藉", "安抚", "抚慰", "安慰", "鼓励"],
    "痛苦": ["煎熬", "难忍", "苦楚", "痛楚", "难过"],
    "慷慨": ["大方", "宽容", "豪爽", "慷慨解囊", "热心"],
    "吝啬": ["小气", "吝啬鬼", "抠门", "抠抠搜搜", "节俭"],
    "强烈": ["剧烈", "激烈", "猛烈", "强大", "明显"],
    "微弱": ["微小", "弱", "细微", "低微", "微不足道"],
    "温和": ["柔和", "温柔", "亲切", "文雅", "和蔼"],
    "严厉": ["严格", "苛刻", "严肃", "凶狠", "严酷"],
    "轻快": ["轻松", "愉悦", "快活", "活泼", "明快"],
    "缓慢": ["迟缓", "慢悠悠", "缓慢无比", "迟迟", "拖沓"],
    "快速": ["迅速", "快捷", "飞快", "敏捷", "急速"],
    "敏感": ["灵敏", "细腻", "敏锐", "敏感性", "细心"],
    "迟钝": ["愚笨", "迟缓", "钝感", "愚钝", "笨拙"],
    "丰富": ["充实", "多样", "多元", "博大", "富饶"],
    "单一": ["单调", "简单", "孤立", "单纯", "单一化"],
    "古老": ["古旧", "陈旧", "悠久", "传统", "老旧"],
    "现代": ["当代", "新型", "时尚", "现代化", "新时代"],
    "严肃": ["庄重", "认真", "郑重", "严谨", "谨慎"],
    "轻松": ["自在", "舒适", "无忧", "无压力", "愉快"],
    "深刻": ["深远", "深奥", "复杂", "独特", "洞察"],
    "浅薄": ["肤浅", "浅显", "浅薄无知", "不深刻", "浅陋"],
    "高兴": ["愉快", "兴奋", "开心", "喜悦", "欢快"],
    "惋惜": ["可惜", "遗憾", "哀叹", "惋惜", "伤感"],
    "羞耻": ["羞愧", "耻辱", "愧疚", "耻感", "丢脸"],
    "hello": ["hello", "hi", "good morning", "hey", "greetings"],
    "thank you": ["thanks", "grateful", "appreciate", "thank you", "many thanks"],
    "goodbye": ["bye", "farewell", "see you", "goodbye", "parting"],
    "question": ["doubt", "inquiry", "challenge", "problem", "confusion"],
    "answer": ["solution", "response", "reply", "answer", "response"],
    "how": ["how", "how to", "how to do", "how to handle", "how to deal"],
    "today": ["today", "this day", "this morning", "today's", "the day"],
    "eat": ["eat", "have a meal", "dine", "have food", "eat food"],
    "like": ["prefer", "like", "hobby", "fond of", "love"],
    "study": ["study", "self-study", "learn", "pursue knowledge", "education"],
    "work": ["task", "job", "work", "position", "role"],
    "place": ["location", "area", "place", "venue", "region"],
    "time": ["time", "time period", "period", "moment", "timing"],
    "fast": ["rapid", "quick", "swift", "flying", "speedy"],
    "slow": ["slow", "sluggish", "gradual", "leisurely", "dragging"],
    "beautiful": ["beautiful", "good looking", "charming", "graceful", "bright"],
    "happy": ["joyful", "cheerful", "happy", "content", "delighted"],
    "excited": ["joyous", "excited", "cheerful", "thrilled", "ecstatic"],
    "home": ["residence", "household", "home", "house", "dwelling"],
    "love": ["love", "fond of", "like", "adore", "affection"],
    "phone": ["telephone", "smartphone", "mobile phone", "smart device", "cell phone"],
    "car": ["vehicle", "car", "automobile", "ride", "transport"],
    "movie": ["film", "picture", "movie", "movie picture", "cinema"],
    "music": ["song", "melody", "tune", "sound", "music note"],
    "book": ["book", "literature", "publication", "text", "material"],
    "high": ["height", "tall", "lofty", "towering", "elevated"],
    "low": ["short", "low", "bottom", "shallow", "depressed"],
    "tomorrow": ["tomorrow", "next day", "the following day", "coming day", "tomorrow morning"],
    "last night": ["last night", "the night before", "last evening", "previous night", "the night before"],
    "hobby": ["interest", "preference", "hobby", "liking", "passion"],
    "good": ["good", "excellent", "nice", "outstanding", "great"],
    "bad": ["bad", "poor", "unsatisfactory", "not good", "inferior"],
    "nice": ["beautiful", "charming", "attractive", "lovely", "graceful"],
    "terrible": ["terrible", "bad", "awful", "horrible", "disastrous"],
    "joyful": ["happy", "delighted", "cheerful", "excited", "satisfied"],
    "worried": ["worried", "anxious", "concerned", "uneasy", "nervous"],
    "angry": ["angry", "unhappy", "furious", "outraged", "irritated"],
    "cold": ["cold", "chilly", "cool", "freezing", "frosty"],
    "hot": ["hot", "high temperature", "scorching", "burning", "sweltering"],
    "quick": ["rapid", "swift", "quick", "speedy", "fast"],
    "energy": ["energy", "vitality", "spirit", "dynamism", "power"],
    "quiet": ["quiet", "peaceful", "still", "calm", "serene"],
    "noisy": ["noisy", "loud", "chaotic", "disorderly", "clamorous"],
    "convenient": ["convenient", "easy", "simple", "comfortable", "handy"],
    "complicated": ["difficult", "complicated", "troublesome", "complex", "intricate"],
    "simple": ["simple", "easy", "straightforward", "direct", "basic"],
    "easy": ["easy", "simple", "effortless", "understandable", "straightforward"],
    "difficult": ["difficult", "hard", "challenging", "tough", "strenuous"],
    "interesting": ["interesting", "fun", "amusing", "engaging", "appealing"],
    "boring": ["boring", "dull", "tedious", "monotonous", "uninteresting"],
    "happy": ["happy", "cheerful", "joyful", "excited", "delighted"],
    "fearful": ["afraid", "panic", "terrified", "scared", "fearful"],
    "trust": ["believe", "depend", "rely", "trust", "confide"],
    "scared": ["fearful", "frightened", "timid", "terrified", "afraid"],
    "lucky": ["lucky", "fortunate", "blessed", "lucky", "successful"],
    "failure": ["failure", "setback", "loss", "mistake", "collapse"],
    "success": ["success", "victory", "achievement", "triumph", "accomplishment"],
    "making money": ["earning", "profit", "making money", "income", "earning revenue"],
    "spending money": ["spending", "expenditure", "spending money", "wasting", "payment"],
    "effort": ["effort", "striving", "working hard", "trying", "diligence"],
    "lazy": ["lazy", "slothful", "unproductive", "idle", "lax"],
    "smart": ["clever", "intelligent", "wise", "bright", "sharp"],
    "dumb": ["dull", "stupid", "unintelligent", "silly", "dumb"],
    "cooperation": ["collaboration", "teamwork", "coordination", "joint effort", "alliance"],
    "competition": ["competition", "contest", "rivalry", "race", "struggle"],
    "change": ["change", "alteration", "shift", "transformation", "modification"],
    "stability": ["stability", "solid", "steady", "reliable", "strong"],
    "courage": ["bravery", "guts", "courage", "valor", "daring"],
    "wisdom": ["wisdom", "insight", "intelligence", "cleverness", "knowledge"],
    "kindness": ["benevolence", "goodness", "generosity", "compassion", "gentleness"],
    "freedom": ["freedom", "independence", "liberty", "autonomy", "self-determination"],
    "poverty": ["poverty", "poor", "destitute", "impoverished", "needy"],
    "wealth": ["wealth", "rich", "prosperous", "affluent", "wealthy"],
    "brave": ["brave", "bold", "courageous", "fearless", "daring"],
    "creation": ["creation", "innovation", "invention", "production", "design"],
    "persistence": ["persistence", "resilience", "endurance", "staying power", "determination"],
    "doubt": ["doubt", "uncertainty", "question", "confusion", "misunderstanding"],
    "hope": ["hope", "wish", "expectation", "longing", "desire"],
    "cry": ["cry", "weep", "sorrow", "lament", "weep aloud"],
    "laugh": ["smile", "laughter", "chuckle", "giggle", "laugh out loud"],
    "happiness": ["joy", "bliss", "happiness", "contentment", "cheerfulness"],
    "lonely": ["lonely", "isolated", "solitary", "empty", "alone"],
    "patience": ["patience", "endurance", "persistence", "tolerance", "composure"],
    "pain": ["pain", "suffering", "ache", "torment", "anguish"],
    "hope": ["hope", "wish", "expectation", "desire", "anticipation"],
    "trouble": ["trouble", "confusion", "difficulty", "problem", "distress"],
    "responsibility": ["duty", "task", "responsibility", "obligation", "commitment"],
    "问题": ["疑问", "询问", "挑战", "难题", "困惑"],
    "答案": ["解答", "回复", "答复", "回应", "解决方案"],
    "怎么样": ["如何", "怎么样", "如何做", "如何应对", "如何处理"]
}

# 自定义位置编码层
class PositionalEncoding(Layer):
    def __init__(self, max_len, model_dim):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.model_dim = model_dim

    def call(self, inputs):
        position = np.arange(self.max_len)[:, np.newaxis]  # 获取位置编码
        div_term = np.exp(np.arange(0, self.model_dim, 2) * -(np.log(10000.0) / self.model_dim))
        pe = np.zeros((self.max_len, self.model_dim))
        pe[:, 0::2] = np.sin(position * div_term)  # 偶数维度使用sin
        pe[:, 1::2] = np.cos(position * div_term)  # 奇数维度使用cos
        pe = pe[np.newaxis, ...]
        return inputs + tf.cast(pe, tf.float32)  # 输入加上位置编码

# 语言模型类
class LanguageModel:
    def __init__(self, vocab_size=10000, max_seq_length=20, data_file='train_data.json', model_file='model/.Ravena-LLM_Model.h5', tokenizer_file='model/tokenizer.json', faq_file='model/.faq_data.pkl'):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.data_file = data_file
        self.model_file = model_file.replace('.h5', '.keras')  # 模型文件保存路径
        self.tokenizer_file = tokenizer_file  # Tokenizer 文件路径
        self.faq_file = faq_file  # FAQ 文件路径
        self.tokenizer = None
        self.model = self.build_model()  # 构建模型
        
        self.faq_data = {}  
        self.load_data()  # 加载训练数据
        self.previous_answers = set()
        self.is_trained = False
        self.faq_data = self.load_faq_data()  # 加载FAQ数据

    def load_faq_data(self):
        if os.path.exists(self.faq_file):  # 如果FAQ数据文件存在，加载
            with open(self.faq_file, 'rb') as f:
                faq_data = pickle.load(f)
        else:
            faq_data = {}  # 否则初始化为空字典
        return faq_data

    def save_faq_data(self):
        with open(self.faq_file, 'wb') as f:
            pickle.dump(self.faq_data, f)  # 保存FAQ数据

    def load_data(self):
        try:
            with open(self.data_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.data = data
        except FileNotFoundError:
            self.data = {"data": []}  # 如果数据文件不存在，初始化为空字典

        questions = [item['question'] for item in self.data['data']]
        answers = [item['answer'] for item in self.data['data']]

        # 扩充FAQ数据
        for q, a in zip(questions, answers):
            if q not in self.faq_data:  
                self.faq_data[q] = [a]  
            else:
                self.faq_data[q].append(a)  

        self.save_faq_data()  # 保存FAQ数据

        # 数据增强
        questions = self.augment_data(questions)
        answers = self.augment_data(answers)

        # 分词处理
        questions = [" ".join(jieba.cut(q)) for q in questions]
        answers = [" ".join(jieba.cut(a)) for a in answers]

        # 如果Tokenizer文件不存在，则训练并保存Tokenizer
        if not os.path.exists(self.tokenizer_file):
            self.tokenizer = Tokenizer(num_words=self.vocab_size)
            self.tokenizer.fit_on_texts(questions + answers)

            with open(self.tokenizer_file, 'w', encoding='utf-8') as f:
                json.dump(self.tokenizer.to_json(), f, ensure_ascii=False, indent=4)
        else:
            # 如果Tokenizer文件已存在，直接加载
            with open(self.tokenizer_file, 'r', encoding='utf-8') as f:
                tokenizer_json = json.load(f)
                self.tokenizer = tokenizer_from_json(tokenizer_json)

        # 转换为序列，并进行填充
        self.question_sequences = pad_sequences(self.tokenizer.texts_to_sequences(questions), maxlen=self.max_seq_length)
        self.answer_sequences = [self.tokenizer.texts_to_sequences([a])[0] for a in answers]
        self.answer_sequences = np.array([seq[0] for seq in self.answer_sequences])

    def augment_data(self, data):
        augmented_data = []
        for item in data:
            words = item.split()
            random.shuffle(words)  # 打乱单词顺序

            if random.random() > 0.5:
                words.append(random.choice(words))  # 随机添加单词
            if len(words) > 2 and random.random() > 0.5:
                words.remove(random.choice(words))  # 随机删除单词

            augmented_data.append(' '.join(words))
        return augmented_data

    def build_model(self):
        input_layer = Input(shape=(self.max_seq_length,))
        embedding_layer = Embedding(self.vocab_size, 128)(input_layer)

        # 添加位置编码层
        pos_encoding = PositionalEncoding(self.max_seq_length, 128)(embedding_layer)
        x = pos_encoding
        for _ in range(120):  # 增加 Transformer 层数为 120 层
            attention = MultiHeadAttention(num_heads=8, key_dim=128)(x, x)
            attention = LayerNormalization()(attention)
            attention = Dropout(0.1)(attention)
            x = attention

        # 池化与输出层
        pooling = GlobalAveragePooling1D()(x)
        dropout = Dropout(0.5)(pooling)
        output_layer = Dense(self.vocab_size, activation='softmax')(dropout)

        model = Model(inputs=input_layer, outputs=output_layer)

        optimizer = Adam(learning_rate=0.0001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def train(self, epochs=100, batch_size=64):
        print(Fore.GREEN + "开始训练模型...")
        self.model.fit(self.question_sequences, self.answer_sequences, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1)
        self.is_trained = True
        print(Fore.GREEN + "训练完成！")

    def generate_answer(self, input_text, max_length=20, temperature=0.7, top_p=0.9):
        """生成回答，并模拟RLHF（人类反馈强化学习）和CoT（思维链）推理"""
        if input_text in self.faq_data:
            time.sleep(random.uniform(1, 3))  # 模拟响应延迟
            answers = self.faq_data[input_text]
            selected_answer = random.choice(answers)
            selected_answer = self.replace_synonyms(selected_answer)  # 替换近义词
            return self.format_text(selected_answer)

        seq = pad_sequences(self.tokenizer.texts_to_sequences([input_text]), maxlen=self.max_seq_length)
        generated_seq = list(seq[0])  # 转换为生成序列
        generated_text = ''  # 初始化生成文本

        # 逐步生成答案，模拟思维链推理
        for step in range(max_length):
            if len(generated_seq) > self.max_seq_length:
                generated_seq = generated_seq[-self.max_seq_length:]  # 保持序列长度不超过最大长度

            # 预测下一个词的概率分布
            pred = self.model.predict(np.array([generated_seq]), verbose=0)
            pred = np.log(pred) / temperature  # 使用温度进行平滑
            pred = np.exp(pred) / np.sum(np.exp(pred))  # 计算softmax概率

            # 选取Top-p的词作为候选
            sorted_indices = np.argsort(pred[0])[::-1]  # 从高到低排序
            cumulative_probs = np.cumsum(pred[0][sorted_indices])  # 计算累计概率
            top_p_indices = sorted_indices[cumulative_probs <= top_p]  # 选择累计概率小于top_p的索引

            next_word_index = np.random.choice(top_p_indices)  # 从中选择下一个词
            generated_seq.append(next_word_index)  # 将新词加入生成序列

            if next_word_index == 0:  # 如果生成的是词汇表中的[PAD]标记，则停止生成
                break

            # 获取生成词对应的文本
            word = self.tokenizer.index_word.get(next_word_index, '')
            generated_text += word + ' '

        # 清理文本并格式化输出
        generated_text = self.clean_text(generated_text)

        # 将新的生成结果添加到FAQ数据中，便于后续参考
        self.faq_data[input_text] = [generated_text]
        self.save_faq_data()

        return self.format_text(generated_text)

    def clean_text(self, text):
        """清理生成文本，去除多余空格等"""
        cleaned_text = ' '.join(text.split())
        cleaned_text = cleaned_text.replace("  ", " ").strip()
        return cleaned_text

    def format_text(self, text):
        """格式化文本，去除不必要的空格和标点"""
        text = re.sub(r'\s([?.!,":;(){}])', r'\1', text)  # 去除标点前多余的空格
        text = re.sub(r'\n', ' ', text)  # 去除换行符
        text = text.strip()
        return text

    def replace_synonyms(self, text):
        """使用近义词替换文本中的词汇"""
        words = text.split()
        replaced_words = []
        for word in words:
            if word in synonyms_dict:
                replaced_word = random.choice(synonyms_dict[word])
                replaced_words.append(replaced_word)
            else:
                replaced_words.append(word)
        return ' '.join(replaced_words)

