'''
アルゴリズム部分のみ弄る
'''

import re
from heapq import heapify, heappush, heappop
from typing import List, Tuple
from enum import Enum
from datetime import date

import numpy as np
from pydantic import BaseModel
import pandas as pd
import numpy as np

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
    # 出力df
    first_concat = 0
    # 希望条件数
    hope_num = 2
    person_table = _read_target_table()
    inst = _read_timetable()
    target = Person.from_dataframe(person_table)
    inst_t = _transform_inst_table(inst)
    inst_cap_sum = inst_t[Col.CAP].sum(axis=0)
    determined_target_list = []
    # 優先度つきのキューに変換
    heapify(target)
    # 手動割り振りリスト
    human_power_list = []
    while target:
        print('rest: ', len(target))
        person = heappop(target)
        req = person.priority
        # 希望条件を減らしていく
        for i in range(hope_num, -1, -1):
            start_check_len = len(person.check)
            end_check_len = start_check_len + 1
            print(i)
            if start_check_len == 0:
                break

            rest_priority = req[:i]
            # 同じ希望条件数で複数回割り振りを行いたい
            while start_check_len != end_check_len:
                start_check_len = len(person.check)
                available_inst = inst_t[inst_t[Col.TYPE].isin(person.check) & (inst_t[Col.CAP] > 0)]
                # 健診を受けられる会場がない場合
                if Req.INST in rest_priority:
                    available_inst = available_inst[available_inst[Col.NAME].isin(person.regions)]
                if Req.WEEKDAY in rest_priority:
                    available_inst = available_inst[available_inst[Col.WEEKDAY].isin(person.weekdays)]
                if Req.MONTH in rest_priority:
                    available_inst = available_inst[available_inst[Col.MONTH].isin(person.months)]
                if (available_inst.shape[0] == 0) & (len(rest_priority) == 0):
                    human_power_list.append(person)
                    break
                if available_inst.shape[0] == 0:
                    break
                if available_inst.groupby(by=[Col.NAME, Col.DATE])[Col.TYPE].nunique().max() < len(person.check):
                    break

                # 健診項目が1つの場合は1番時間が早い場所に申し込む
                if start_check_len == 1:
                    best_inst = available_inst.sort_values(by=[Col.HEAD])[Col.NAME].iloc[0]
                    best_date = available_inst.sort_values(by=[Col.HEAD])[Col.DATE].iloc[0]
                    best_med_list = available_inst.sort_values(by=[Col.HEAD])[Col.TYPE].iloc[0]
                    best_time_list = available_inst.sort_values(by=[Col.HEAD])[Col.HEAD].iloc[0]
                    _last = available_inst.sort_values(by=[Col.HEAD])[Col.LAST].iloc[0]
                else:
                    # 同日で最も多くの検診を受けられる場所と日時を探している。
                    inst_list = []
                    med_list = []
                    counter_set = 0
                    for v, k in zip(available_inst.groupby(by=[Col.NAME, Col.DATE])[Col.TYPE].unique(),
                                    available_inst.groupby(by=[Col.NAME, Col.DATE])[Col.TYPE].unique().index):
                        counter = 0
                        for l in v:
                            if l in person.check:
                                counter += 1
                        if counter > counter_set:
                            inst_list = []
                            med_list = []
                            inst_list.append(k)
                            med_list.append(v)
                            counter_set = counter
                        elif counter == counter_set:
                            inst_list.append(k)
                            med_list.append(v)
                            assert len(inst_list) == len(med_list)

                    # なるべく待ち時間が少ない医療機関を選択
                    best_wait_time = 1000000
                    best_time_list = np.nan
                    best_inst = np.nan
                    best_date = np.nan
                    best_med_list = np.nan
                    for p in range(0, len(inst_list)):
                        _name, _date = inst_list[p]
                        _order = med_list[p]
                        # 希望に合わせて絞ったデータフレーム
                        available_inst_sort = available_inst[
                            (available_inst[Col.NAME] == _name) & (available_inst[Col.DATE] == _date)]
                        early_wait_time = available_inst_sort.groupby(by=[Col.TYPE])[Col.HEAD].min().diff().abs().max()
                        early_time = available_inst_sort.groupby(by=[Col.TYPE])[Col.HEAD].min()
                        late_wait_time = available_inst_sort.groupby(by=[Col.TYPE])[Col.HEAD].max().diff().abs().max()
                        late_time = available_inst_sort.groupby(by=[Col.TYPE])[Col.HEAD].max()
                        min_wait_time = min(early_wait_time, late_wait_time)
                        if early_wait_time <= late_wait_time:
                            best_time = early_time
                        else:
                            best_time = late_time
                        if min_wait_time < best_wait_time:
                            best_wait_time = min_wait_time
                            best_inst = _name
                            best_date = _date
                            best_med_list = best_time.index
                            best_time_list = best_time

                # 受診日にフラグ立て
                # 健診項目が1つの時
                if type(best_time_list) == np.int64:
                    available_inst.loc[
                        (available_inst[Col.NAME] == best_inst) & (available_inst[Col.DATE] == best_date) & (
                                available_inst[Col.TYPE] == best_med_list) & (
                                available_inst[Col.HEAD] == best_time_list), 'extract'] = 1
                    assert available_inst["extract"].sum(axis=0) == 1

                    # 割り振り情報を追加
                    output_df_raw = available_inst[available_inst['extract'] == 1].copy()
                    output_df_raw["id"] = person.id
                    if first_concat == 0:
                        output_df = output_df_raw.copy()
                        first_concat += 1
                    else:
                        output_df = pd.concat([output_df, output_df_raw], axis=0)

                    # tt = Timetable(
                    #     name=best_inst,
                    #     date=best_date,
                    #     type=_type,
                    #     head=_head,
                    #     last=_last
                    # )
                    person.check.remove(best_med_list)

                    # 健診機関の健診種別と開始時間・終了時間の情報をPersonに追加、健診機関のcapacityを1減らす
                    # person.append_timetable(tt)
                    inst_t.loc[(inst_t[Col.NAME] == best_inst) & (inst_t[Col.DATE] == best_date) &
                               (inst_t[Col.TYPE] == best_med_list) & (inst_t[Col.HEAD] == best_time_list) &
                               (inst_t[Col.LAST] == _last), Col.CAP] -= 1

                    end_check_len = len(person.check)
                    if end_check_len == 0:
                        break
                # 健診項目が2つ以上の時
                else:
                    for m, n in zip(best_med_list, best_time_list):
                        available_inst.loc[
                            (available_inst[Col.NAME] == best_inst) & (available_inst[Col.DATE] == best_date) & (
                                    available_inst[Col.TYPE] == m) & (available_inst[Col.HEAD] == n), 'extract'] = 1
                    assert available_inst["extract"].sum(axis=0) == len(best_time_list)

                    # 割り振り情報を追加
                    output_df_raw = available_inst[available_inst['extract'] == 1].copy()
                    output_df_raw["id"] = person.id
                    if first_concat == 0:
                        output_df = output_df_raw.copy()
                        first_concat += 1
                    else:
                        output_df = pd.concat([output_df, output_df_raw], axis=0)

                    for h in range(len(best_time_list)):
                        _head = output_df_raw[Col.HEAD].iloc[h]
                        _last = output_df_raw[Col.LAST].iloc[h]
                        _type = output_df_raw[Col.TYPE].iloc[h]
                        # tt = Timetable(
                        #     name=best_inst,
                        #     date=best_date,
                        #     type=_type,
                        #     head=_head,
                        #     last=_last
                        # )
                        person.check.remove(_type)

                        # 健診機関の健診種別と開始時間・終了時間の情報をPersonに追加、健診機関のcapacityを1減らす
                        # person.append_timetable(tt)
                        inst_t.loc[(inst_t[Col.NAME] == best_inst) & (inst_t[Col.DATE] == best_date) &
                                   (inst_t[Col.TYPE] == _type) & (inst_t[Col.HEAD] == _head) &
                                   (inst_t[Col.LAST] == _last), Col.CAP] -= 1

                    end_check_len = len(person.check)
                    if end_check_len == 0:
                        break
    print(inst_cap_sum)
    print(inst_t[Col.CAP].sum(axis=0))
    print(output_df.shape[0])
    print(output_df.shape[0] + inst_t[Col.CAP].sum(axis=0))


main()

# for i in range(10):
#     abc = [11, 12, 13, 14, 15]
#     print(i)
#     while True:
#         abcd = abc.pop()
#         print(abcd)
#         if abcd == 13:
#             break
