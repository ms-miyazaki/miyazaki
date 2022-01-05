'''
アルゴリズム部分のみ弄る

各dfの説明
    ・予約者情報df：person_table
    ・キャパシティ情報df：inst_t
    ・予約結果df：output_df

作成部分
    ・244行目：宮﨑作成関数以下

アルゴリズム
    ・貪欲方で割り振っています。

'''

import re
import shlex
from heapq import heapify, heappush, heappop
from typing import List, Tuple
from enum import Enum
from datetime import date

import dascan
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
    'アスタホール': '大利根地域',
    'アスターホール': '大利根地域',
    '希望なし': ''
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
    ID = '宛名番号'
    NAME = '受診会場'
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


'''
宮﨑作成関数
'''


def allocation_checker(df, raw_df):
    '''
    raw_dfにおいて指定されている「月」、「曜日」、「受診会場」...等を正しく割り振れているかの確認
    dfとraw_dfを照らし合わせ、df列の「希望_月」、「希望_曜日」...等の列にフラグ（0or1）を立てる。
    '''
    if str(raw_df[raw_df[Col.ID] == int(df[Col.ID].unique())][Col.WEEKDAY]) == "希望なし":
        df[f"希望_{Col.WEEKDAY}"] = 1
    elif WEEKDAYS[int(df[Col.WEEKDAY].unique())] in \
            raw_df[raw_df[Col.ID] == int(df[Col.ID].unique())][Col.WEEKDAY].iloc[
                -1].split("・"):
        df[f"希望_{Col.WEEKDAY}"] = 1
    else:
        df[f"希望_{Col.WEEKDAY}"] = 0

    if str(raw_df[raw_df[Col.ID] == int(df[Col.ID].unique())][Col.NAME]) == "希望なし":
        df[f"希望_{Col.NAME}"] = 1
    elif df[Col.NAME].iloc[-1] in raw_df[raw_df[Col.ID] == int(df[Col.ID].unique())][Col.NAME].iloc[-1].split("・"):
        df[f"希望_{Col.NAME}"] = 1
    else:
        df[f"希望_{Col.NAME}"] = 0

    if str(raw_df[raw_df[Col.ID] == int(df[Col.ID].unique())][Col.MONTH]) == "希望なし":
        df[f"希望_{Col.MONTH}"] = 1
    elif str(df[Col.MONTH].iloc[-1]) in raw_df[raw_df[Col.ID] == int(df[Col.ID].unique())][Col.MONTH].iloc[-1].replace(
            "月",
            "").split(
        "・"):
        df[f"希望_{Col.MONTH}"] = 1
    else:
        df[f"希望_{Col.MONTH}"] = 0

    return df


def hope_checker(df, df_raw):
    '''
    raw_dfの優先順位1,2を満たせているかの確認。
    df列の「希望_XX」列のフラグ（0or1）を参照してdf列の「結果_優先順位1」、「結果_優先順位2」にフラグ（0or1）立て
    '''
    hope_dict = {"月": "希望_月",
                 "会場": "希望_受診会場",
                 "曜日": "希望_曜日",
                 "健(検)診項目": "希望_健診項目"}

    df["結果_優先順位1"] = 0
    df["結果_優先順位2"] = 0

    for raw_num in range(len(df)):
        id = df.iloc[raw_num][Col.ID]
        if (df_raw[df_raw[Col.ID] == id]["優先順位1"].iloc[-1]) != (df_raw[df_raw[Col.ID] == id]["優先順位1"].iloc[-1]):
            df.iloc[raw_num, 15] = 1
        if (df_raw[df_raw[Col.ID] == id]["優先順位2"].iloc[-1]) != (df_raw[df_raw[Col.ID] == id]["優先順位2"].iloc[-1]):
            df.iloc[raw_num, 16] = 1
        for col in hope_dict.keys():
            if col == "健(検)診項目":
                if (df[df[Col.ID] == id].groupby(by=[Col.ID])[Col.DATE].nunique().iloc[-1] == 1) & (
                        df_raw[df_raw[Col.ID] == id]["優先順位1"].iloc[-1] == col):
                    df.iloc[raw_num, 15] = 1
            elif col == "健(検)診項目":
                if (df[df[Col.ID] == id].groupby(by=[Col.ID])[Col.DATE].nunique().iloc[-1] == 1) & (
                        df_raw[df_raw[Col.ID] == id]["優先順位2"].iloc[-1] == col):
                    df.iloc[raw_num, 16] = 1
            else:
                if (df_raw[df_raw[Col.ID] == id]["優先順位1"].iloc[-1] == col) & (df.iloc[raw_num][hope_dict[col]] == 1):
                    df.iloc[raw_num, 15] = 1
                if (df_raw[df_raw[Col.ID] == id]["優先順位2"].iloc[-1] == col) & (df.iloc[raw_num][hope_dict[col]] == 1):
                    df.iloc[raw_num, 16] = 1


def Day_checker(df):
    '''
    下記2つの確認。assertが出なければok
    ・1日1会場で検診を行えているかの確認
    ・健診の日程が2日以内になっているかの確認
    '''
    assert all(df.groupby(by=[id, Col.DATE])[Col.NAME].nunique() == 1)
    assert df.groupby(by=[Col.ID])[Col.DATE].nunique().max() <= 2
    assert df.groupby(by=[Col.ID])[Col.REGION].nunique().max() <= 2


def Name_checker(df, raw_df):
    '''
    希望していて資格のある検診を過不足なく受けられているかの確認。assertが出なければok
    df_rawの「資格あり_XX」列のフラグ（0or1）を元に、1のフラグがある健診は受けられるか、0のフラグがある検診を受けてしまっていないか判定
    '''
    target_columns = ['資格あり_特定健診', '資格あり_胃がん', '資格あり_肺がん', '資格あり_大腸がん', '資格あり_乳がん', '資格あり_子宮がん', '資格あり_前立腺がん']
    for ids in df[Col.ID].unique():
        for columns in target_columns:
            if raw_df[(raw_df[Col.ID] == ids) & (raw_df[columns] == 1)].shape[0] == 1:
                assert columns.replace("資格あり_", "") in \
                       df[df[Col.ID] == ids].groupby(by=[Col.ID])[Col.TYPE].unique().iloc[-1].tolist()
            else:
                assert columns.replace("資格あり_", "") not in \
                       df[df[Col.ID] == ids].groupby(by=[Col.ID])[Col.TYPE].unique().iloc[-1].tolist()


def add_region(x):
    _list = []
    for inst_name in x.split('・'):
        _list.append(INST_REGIONS[inst_name])

    return '・'.join(set(_list))


def remove_df(df, rem_list):
    print(df.shape)
    while rem_list:
        rem = heappop(rem_list)
        df = df[df[Col.ID] != rem.id]
    print(df.shape)
    return df


def main():
    first_concat = 0  # concat元のdfになるか
    person_table = _read_target_table()

    person_table = person_table[~person_table['ラスト ネーム'].isin(['大塚', '江森'])]  # 除外フラグを削除
    person_table = person_table[~person_table['No.'].isin([31, 241])]  # 重複を削除
    assert person_table[person_table.duplicated(subset=[Col.ID], keep=False)].shape[0] == 0
    person_table["希望していて資格ある数"] = person_table[['資格あり_特定健診', '資格あり_胃がん', '資格あり_肺がん', '資格あり_大腸がん', '資格あり_乳がん',
                                                '資格あり_子宮がん', '資格あり_前立腺がん']].sum(axis=1)  # 受ける健診数を更新
    person_table['希望地域'] = person_table['受診会場'].apply(add_region)
    inst = _read_timetable()
    target = Person.from_dataframe(person_table)
    inst_t = _transform_inst_table(inst)
    inst_t = inst_t[
        [Col.NAME, Col.TYPE, Col.CAP, Col.DATE, Col.MONTH, Col.WEEKDAY, Col.TIME, Col.HEAD, Col.LAST, Col.REGION]]
    inst_cap_sum = inst_t[Col.CAP].sum(axis=0)
    # 優先度つきのキューに変換
    heapify(target)
    # 手動割り振りリスト
    human_power_list = []
    while target:
        print(f'----------残りの人数：{len(target)}----------')
        person = heappop(target)
        print(f'id：{person.id}')
        print(f'健診項目数：{len(person.check)}')
        already_date = []  # 同日に2会場以上での受診を防ぐために使用
        target_region = []  # 同じ地域で受診するために使用
        req = person.priority  # 希望条件
        for i in range(len(req), -1, -1):
            '''
            なるべく多くの条件を満たせる事がベスト
            該当する医療機関がない場合は条件を減らしていく
            '''
            start_check_len = len(person.check)
            end_check_len = start_check_len + 1
            print(f'希望項目数：{i}')
            if start_check_len == 0:
                break
            if len(target_region) >= 2:
                human_power_list.append(person)
                break
            rest_priority = req[:i]
            while start_check_len != end_check_len:
                '''
                条件に合わせて医療機関を絞っていく
                '''
                start_check_len = len(person.check)
                if target_region:
                    available_inst = inst_t[
                        inst_t[Col.TYPE].isin(person.check) & (inst_t[Col.CAP] > 0) & ~inst_t[Col.DATE].isin(
                            already_date) & (inst_t[Col.REGION].isin(target_region))]
                else:
                    available_inst = inst_t[
                        inst_t[Col.TYPE].isin(person.check) & (inst_t[Col.CAP] > 0) & ~inst_t[Col.DATE].isin(
                            already_date)]
                if Req.INST in rest_priority:
                    available_inst = available_inst[available_inst[Col.NAME].isin(person.regions)]
                if Req.WEEKDAY in rest_priority:
                    available_inst = available_inst[available_inst[Col.WEEKDAY].isin(person.weekdays)]
                if Req.MONTH in rest_priority:
                    available_inst = available_inst[available_inst[Col.MONTH].isin(person.months)]

                # 健診を受けられる会場がない場合
                if (available_inst.shape[0] == 0) & (len(rest_priority) == 0):
                    human_power_list.append(person)
                    break
                # if (available_inst.groupby(by=[Col.NAME, Col.DATE])[Col.TYPE].nunique().max() < len(person.check)) & (
                #         len(rest_priority) == 0):
                #     human_power_list.append(person)
                #     break
                if available_inst.shape[0] == 0:
                    break
                # if available_inst.groupby(by=[Col.NAME, Col.DATE])[Col.TYPE].nunique().max() < len(person.check):
                #     break
                if ('大腸がん' in available_inst[Col.TYPE].unique()) & (
                        available_inst.groupby(by=[Col.NAME, Col.DATE])[Col.TYPE].nunique().max() == 1):
                    if len(rest_priority) == 0:
                        human_power_list.append(person)
                    break
                if ('前立腺がん' in available_inst[Col.TYPE].unique()) & ('特定健診' not in available_inst[Col.TYPE].unique()):
                    if len(rest_priority) == 0:
                        human_power_list.append(person)
                    break
                # 健診項目が1つの場合は1番時間が早い場所に申し込む
                if (start_check_len == 1) or (
                        available_inst.groupby(by=[Col.NAME, Col.DATE])[Col.TYPE].nunique().max() == 1):
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
                    output_df_raw[Col.ID] = person.id
                    allocation_checker(output_df_raw, person_table)
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
                    already_date.append(best_date)
                    target_region.append(INST_REGIONS[best_inst])

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
                    output_df_raw[Col.ID] = person.id
                    output_df_raw = allocation_checker(output_df_raw, person_table)
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
                        already_date.append(best_date)

                        # 健診機関の健診種別と開始時間・終了時間の情報をPersonに追加、健診機関のcapacityを1減らす
                        # person.append_timetable(tt)
                        inst_t.loc[(inst_t[Col.NAME] == best_inst) & (inst_t[Col.DATE] == best_date) &
                                   (inst_t[Col.TYPE] == _type) & (inst_t[Col.HEAD] == _head) &
                                   (inst_t[Col.LAST] == _last), Col.CAP] -= 1
                    target_region.append(INST_REGIONS[best_inst])

                    end_check_len = len(person.check)
                    if end_check_len == 0:
                        break
    dame_list = human_power_list.copy()
    output_df = remove_df(output_df, human_power_list)
    hope_checker(output_df, person_table)
    Day_checker(output_df)
    Name_checker(output_df, person_table)
    output_path = 's3://da-cleaning-lgdata-v1/2021/埼玉県加須市/02.mailing/next_year_booking/99.for_work/training/#宮﨑さん石黒さんトレーニング用/宮﨑output/'
    output_name = '振り分け一覧.csv'
    dascan.to_csv(output_df, output_path + output_name, index=False, encoding='utf-8-sig')
    output_name = '振り分け後キャパシティ.csv'
    dascan.to_csv(inst_t, output_path + output_name, index=False, encoding='utf-8-sig')
    output_name = '予約データ.csv'
    dascan.to_csv(person_table, output_path + output_name, index=False, encoding='utf-8-sig')


main()

# print(f'医療機関キャパシティ合計：{inst_cap_sum}')
# print(f'振り分け後に残ったキャパシティ合計：{inst_t[Col.CAP].sum(axis=0)}')
print(f'希望健診数合計：{person_table["希望していて資格ある数"].sum(axis=0)}')
print(f'振り分け数：{output_df.shape[0]}')
# print(f'振り分け数と残ったキャパシティが合うか：{(output_df.shape[0] + inst_t[Col.CAP].sum(axis=0)) == inst_cap_sum}')
print(f'割り振れなかった人数：{len(dame_list)}')
# print(
#     f'割り振れなかった人数が正しいか：{sum(person_table[~person_table[Col.ID].isin(output_df[Col.ID])]["希望していて資格ある数"] != 0) == len(human_power_list)}')
#
# # print(f'割り振れなかった人数が正しいか{(person_table[~person_table[Col.ID].isin(output_df[Col.ID])].shape[0])}')
# # person_table[~person_table[Col.ID].isin(output_df[Col.ID])]

'''
宮﨑確認用
'''

#
# def agg_check(df1, df2, columns1, columns2):
#     df1.rename(columns={Col.ID: Col.ID}, inplace=True)
#     merged = pd.merge(df1, df2, how="left", on=Col.ID)
#     merged.loc[merged[columns2].isna()] = 0
#     # merged[columns2] = merged[columns2].astype(int)
#     print('件数が一致していないもの')
#     return merged[merged[columns1] != merged[columns2]], merged[merged[columns1] != merged[columns2]][Col.ID]
#
# check, id = agg_check(person_table, output_df.groupby(by=[Col.ID])[Col.NAME].count().reset_index(), "希望していて資格ある数",
#                       Col.NAME)


main()


# for i in range(10):
#     abc = [11, 12, 13, 14, 15]
#     print(i)
#     while True:
#         abcd = abc.pop()
#         print(abcd)
#         if abcd == 13:
#             break


def main():
    person_table = _read_target_table()
    person_table = person_table[~person_table['ラスト ネーム'].isin(['大塚', '江森'])]  # 除外フラグを削除
    person_table = person_table[~person_table['No.'].isin([31, 241])]  # 重複を削除
    assert person_table[person_table.duplicated(subset=[Col.ID], keep=False)].shape[0] == 0
    person_table["希望していて資格ある数"] = person_table[['資格あり_特定健診', '資格あり_胃がん', '資格あり_肺がん', '資格あり_大腸がん', '資格あり_乳がん',
                                                '資格あり_子宮がん', '資格あり_前立腺がん']].sum(axis=1)  # 受ける健診数を更新
    inst = _read_timetable()
    target = Person.from_dataframe(person_table)
    inst_t = _transform_inst_table(inst)
    determined_target_list = []
    # 優先度つきのキューに変換
    heapify(target)
    while target:
        print('rest: ', len(target))
        person = heappop(target)
        req = person.priority
        for i in range(len(req) + 1):
            rest_priority = req[:len(req) - i]
            # 資格が希望している健診を受診できる会場（会場キャパも考慮）に絞る
            available_inst = inst_t[inst_t[Col.TYPE].isin(person.check) & (inst_t[Col.CAP] > 0)]
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
            if available_inst.empty:
                continue
            # available_instの中で最も健診を受けることができる日を抽出している
            _name, _date = available_inst.groupby(by=[Col.NAME, Col.DATE])[Col.TYPE].nunique().idxmax()
            # 上記の健診機関と日付で絞ったデータフレーム
            available_inst = available_inst[(available_inst[Col.NAME] == _name) & (available_inst[Col.DATE] == _date)]
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
divide_df['id'] = divide_df['id'].astype(int)

output_sum_df = output_df.groupby(by=[Col.ID])[Col.TYPE].nuique().reset_index()

check_df = pd.merge(person_table[['宛名番号', '希望していて資格ある数']], divide_df[['id', '健診数']], left_on='宛名番号', right_on='id',
                    how='left')
sum(check_df['希望していて資格ある数'] != check_df['健診数'])
check_df[check_df['希望していて資格ある数'] != check_df['健診数']]
df_1 = person_table[person_table['宛名番号'].isin(check_df[check_df['希望していて資格ある数'] != check_df['健診数']]['宛名番号'])]
