import re
from heapq import heapify, heappush, heappop
from typing import List, Tuple
from enum import Enum
from datetime import date
from pydantic import BaseModel
import pandas as pd

FISCAL_YEAR = 2021
BASE_PATH = "s3://da-cleaning-lgdata-v1/2021/埼玉県加須市/02.mailing/next_year_booking/99.for_work/training/"
TIME_TABLE_PATH = BASE_PATH + '埼玉県加須市_修正後.xlsx'
PERSON_TABLE_PATH = BASE_PATH + 'cleaning_target_df_v2.csv'
REGIONS = ['北川辺地域', '加須地域', '騎西地域', '大利根地域']
INST_REGIONS = {
    '北川辺健康福祉センター': '北川辺地域',
    '加須保健センター': '加須地域',
    'パストラルかぞ': '加須地域',
    '花崎コミュニティセンター': '加須地域',
    '騎西健康福祉センター': '騎西地域',
    '大利根健康福祉センター': '大利根地域',
    'アスタホール': '大利根地域'
}
WEEKDAYS = ['月', '火', '水', '木', '金', '土', '日']


class CheckType(str, Enum):
    GASTRIC = '胃がん'
    LUNG = '肺がん'
    COLON = '大腸がん'
    TOKUTEI = '特定健診'
    BREAST = '乳がん'
    UTERUS = '子宮がん'
    PROSTATE = '前立腺がん'


class Col(str, Enum):
    NAME = '検診会場'
    DATE = '日付'
    TYPE = '検診種別'
    TIME = '時間'
    CAP = '収容人数'
    MONTH = '月'
    WEEKDAY = '曜日'
    HEAD = '開始時間'
    LAST = '終了時間'
    REGION = '地区'


class Req(str, Enum):
    MEDICAL = "健(検)診項目"
    REGION = '地区'
    WEEKDAY = '曜日'
    MONTH = '月'
    INST = '会場'


def _rename_func_for_inst(col: str) -> str:
    if col == '特定健診\n後期健診':
        return '特定健診'
    return col


def _read_timetable():
    df = pd.read_excel(TIME_TABLE_PATH, sheet_name=1, header=[0, 1])
    df = df.rename(columns=_rename_func_for_inst, level=0)
    return df


def _rename_func_for_target(col: str) -> str:
    return col.replace('検診', '')


def _read_target_table():
    df = pd.read_csv(PERSON_TABLE_PATH)
    df = df.rename(columns=_rename_func_for_target)
    return df


def _strtime2int(t: str) -> int:
    """
    >>> _strtime2int("11:15")
    1115
    """
    return int("".join(t.split(":")))


def _parse_time(t_t: str) -> Tuple[int, int]:
    """
    >>> _parse_time("11:15-11:45")
    (1115, 1145)
    """
    t1, t2 = t_t.split('-')
    return _strtime2int(t1), _strtime2int(t2)


def _parse_date(_d: str) -> date:
    _d = re.search(r'[0-9０-９]*/[0-9０-９]*', _d)[0]
    m, d = map(int, _d.split('/'))
    return date(FISCAL_YEAR, m, d)


def _transform_inst_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    1レコードが検診会場・日付でユニークなテーブルを検診会場・日付・検診項目・検診時間でユニークなテーブルに変換する関数
    """
    id_vars = list(df.columns[:2])
    value_vars = [
        col for col in df.columns
        if col[0] in CheckType.__members__.values() and col[1] != '定員'
    ]
    transformed = pd.melt(
        frame=df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=[Col.TYPE, Col.TIME],
        value_name=Col.CAP,
    )
    transformed.columns = [Col.NAME, Col.DATE, Col.TYPE, Col.TIME, Col.CAP]
    transformed[Col.DATE] = transformed[Col.DATE].apply(_parse_date)
    transformed[Col.HEAD] = transformed[Col.TIME].apply(lambda x: _parse_time(x)[0])
    transformed[Col.LAST] = transformed[Col.TIME].apply(lambda x: _parse_time(x)[1])
    transformed[Col.MONTH] = transformed[Col.DATE].apply(lambda x: x.month)
    transformed[Col.WEEKDAY] = transformed[Col.DATE].apply(lambda x: x.weekday())
    transformed[Col.REGION] = transformed[Col.NAME].apply(INST_REGIONS.get)
    return transformed.loc[transformed[Col.CAP] > 0]


def _parse_req_month(s_months: str) -> List[int]:
    if s_months == '希望なし':
        return [i for i in range(1, 13)]
    months = re.findall(r'[0-9０-９]+', s_months)
    return list(map(int, months))


def _parse_req_weekday(s_weekdays: str) -> List[int]:
    if s_weekdays == '希望なし':
        return [i for i in range(7)]
    weekdays = s_weekdays.split('・')
    return [WEEKDAYS.index(d) for d in weekdays if d in WEEKDAYS]


def _parse_req_regions(s_regions: str) -> List[str]:
    if s_regions != s_regions or s_regions == '希望なし':
        return REGIONS
    region_list = [region for region in s_regions.split('; ') if region != '希望なし']
    return [r.split("・") for r in region_list][0]


def _parse_req_priority(p1: str, p2: str, req_regions: List[str]) -> List[Req]:
    priority_list = []
    for p in [p1, p2]:
        # 空欄の場合
        if p != p:
            continue
        if p == '会場' and not (set(req_regions) & set(REGIONS)):
            priority_list.append(Req.INST)
        else:
            priority_list.append(Req(p))
    return priority_list + [r for r in Req.__members__.values() if r not in priority_list]


def _int_time2str(t: int) -> str:
    return "{0}:{1:02}".format(t // 100, t % 100)


class Timetable(BaseModel):
    name: str
    date: date
    type: CheckType
    head: int
    last: int

    def __gt__(self, other):
        if self.date == other.date:
            return self.head > other.head
        return self.date > other.date

    def __str__(self):
        return f"{self.name}\n{self.date} {_int_time2str(self.head)}-{_int_time2str(self.last)}\n{self.type}"


class Person(BaseModel):
    id: int
    regions: List[str]
    months: List[int]
    weekdays: List[int]
    check: List[CheckType]
    priority: List[Req]
    timetable: List[Timetable] = []

    def __gt__(self, other):
        # heapqで検診項目が大きい順に取り出したいので
        return len(self.check) < len(other.check)

    def to_record(self) -> List[str]:
        tts = sorted(self.timetable)
        record = [str(self.id)]
        record += [str(tt) for tt in tts]
        return record

    def append_timetable(self, tt: Timetable):
        self.check.remove(tt.type)
        self.timetable.append(tt)

    @classmethod
    def _from_srs(cls, srs: pd.Series) -> "Person":
        _id = srs['宛名番号']
        regions = _parse_req_regions(srs['受診会場'])
        months = _parse_req_month(srs['月'])
        weekdays = _parse_req_weekday(srs['曜日'])
        check = [c for c in CheckType.__members__.values() if srs['資格あり_' + c] == 1]
        priority = _parse_req_priority(srs['優先順位1'], srs['優先順位2'], regions)
        return Person(
            id=_id,
            regions=regions,
            months=months,
            weekdays=weekdays,
            check=check,
            priority=priority
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List["Person"]:
        return [Person._from_srs(row) for _, row in df.iterrows()]


def main():
    person_table = _read_target_table()
    inst = _read_timetable()
    target = Person.from_dataframe(person_table)
    inst_t = _transform_inst_table(inst)
    determined_target_list = []
    # 優先度つきのキューに変換
    heapify(target)
    # 情報を持てるように変更
    hold_REGION = []
    hold_id = 999999999999999999999999
    # 手動割り振りリスト
    human_power_list = []
    while target:
        print('rest: ', len(target))
        person = heappop(target)
        print(person.id)
        req = person.priority
        if person.id != hold_id:
            print('change!hold')
            hold_id = person.id
            hold_REGION = []
        for i in range(len(req)):
            rest_priority = req[:len(req) - i]
            # 資格が希望している健診を受診できる会場（会場キャパも考慮）に絞る
            if len(hold_REGION) == 0:
                available_inst = inst_t[inst_t[Col.TYPE].isin(person.check) & (inst_t[Col.CAP] > 0)]
            else:
                print(hold_REGION)
                available_inst = inst_t[inst_t[Col.TYPE].isin(person.check) & (inst_t[Col.CAP] > 0) & (inst_t[Col.REGION].isin(hold_REGION))]
            if Req.MEDICAL in rest_priority:
                available_inst = available_inst[available_inst[Col.TYPE].isin(person.check)]
            if Req.INST in rest_priority:
                available_inst = available_inst[available_inst[Col.NAME].isin(person.regions)]
            if Req.REGION in rest_priority:
                available_inst = available_inst[available_inst[Col.REGION].isin(person.regions)]
            if Req.WEEKDAY in rest_priority:
                available_inst = available_inst[available_inst[Col.WEEKDAY].isin(person.weekdays)]
            if Req.MONTH in rest_priority:
                available_inst = available_inst[available_inst[Col.MONTH].isin(person.months)]
            if available_inst.empty & (len(req) == i):
                human_power_list.append(person)
                continue
            if available_inst.empty:
                continue

            # available_instの中で最も健診を受けることができる日を抽出している
            _name, _date = available_inst.groupby(by=[Col.NAME, Col.DATE])[Col.TYPE].nunique().idxmax()
            # 上記の健診機関と日付で絞ったデータフレーム
            available_inst = available_inst[(available_inst[Col.NAME] == _name) & (available_inst[Col.DATE] == _date)]
            if ('大腸がん' in available_inst[Col.TYPE].unique()) & (available_inst[Col.TYPE].nunique() == 1):
                continue
            if ('前立腺がん' in available_inst[Col.TYPE].unique()) & ('特定健診' not in available_inst[Col.TYPE].unique()):
                continue
            # ソートする優先順位順に並び替え（健診項目が少ない順）
            _order = ["乳がん", "子宮がん", "特定健診", "前立腺がん", "胃がん", "肺がん", "大腸がん"]
            available_inst["order"] = available_inst[Col.TYPE].apply(lambda x: _order.index(x) if x in _order else -1)
            available_inst.sort_values(by="order", inplace=True)
            while not available_inst.empty:
                available_inst = available_inst.sort_values(by=Col.HEAD)
                _head = available_inst[Col.HEAD].iloc[0]
                _last = available_inst[Col.LAST].iloc[0]
                _type = available_inst[Col.TYPE].iloc[0]
                tt = Timetable(
                    name=_name,
                    date=_date,
                    type=_type,
                    head=_head,
                    last=_last
                )
                # 健診機関の健診種別と開始時間・終了時間の情報をPersonに追加、健診機関のcapacityを1減らす
                person.append_timetable(tt)
                inst_t.loc[(inst_t[Col.NAME] == _name) & (inst_t[Col.DATE] == _date) &
                           (inst_t[Col.TYPE] == _type) & (inst_t[Col.HEAD] == _head) &
                           (inst_t[Col.LAST] == _last), Col.CAP] -= 1
                # 同一時間内に収めたい場合
                available_inst = available_inst[available_inst[Col.TYPE].isin(person.check)]
                # 時間を変えたい場合
                # available_inst = available_inst[available_inst[Col.TYPE].isin(person.check) &
                #                                 (available_inst[Col.HEAD] >= _last)]
            hold_REGION.append(inst_t.loc[(inst_t[Col.NAME] == _name) & (inst_t[Col.DATE] == _date)][Col.REGION])
            # 健診項目が最優先の人はすべて埋まらなくとも次の処理へ向かう
            # if Req.MEDICAL in rest_priority:
            #     determined_target_list.append(person)
            if not person.check:
                determined_target_list.append(person)
            else:
                heappush(target, person)
            break
    return pd.DataFrame.from_records([p.to_record() for p in determined_target_list]), inst_t


divide_df, inst_df = main()
person_table = _read_target_table()
divide_df.columns = ["id", "d_1", 'd_2', 'd_3', 'd_4', 'd_5', 'd_6']
divide_df['健診数'] = divide_df[["d_1", 'd_2', 'd_3', 'd_4', 'd_5', 'd_6']].notna().sum(axis=1)
divide_df.sort_values(by=['健診数'], inplace=True, ascending=False)

for i in range(10):
    print(i)
    continue
    print('a')

