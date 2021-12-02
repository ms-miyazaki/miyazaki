'''
使用データの理解

'''

import pandas as pd

FISCAL_YEAR = 2021
BASE_PATH = "s3://da-cleaning-lgdata-v1/2021/埼玉県加須市/02.mailing/next_year_booking/99.for_work/training/"
TIME_TABLE_PATH = BASE_PATH + '埼玉県加須市_修正後.xlsx'
PERSON_TABLE_PATH = BASE_PATH + 'cleaning_target_df_v2.csv'

booking_df = pd.read_csv(PERSON_TABLE_PATH)
capacity_df = pd.read_excel(TIME_TABLE_PATH, sheet_name=1, header=[0, 1])


def check_booking():
    booking_df["優先順位1"].value_counts()
    '''
    会場         134
    健(検)診項目     82
    曜日          48
    月           37
    '''
    booking_df["優先順位2"].value_counts()
    '''
    会場         106
    月           63
    曜日          60
    健(検)診項目     59
    '''
    booking_df["月"].value_counts()
    '''
    9月・10月                  91
    5月・6月                   84
    11月・12月                 55
    希望なし                    48
    9月・10月・11月・12月          12
    5月・6月・9月・10月            11
    5月・6月・9月・10月・11月・12月     5
    5月・6月・11月・12月            1
    '''
    booking_df["曜日"].value_counts()
    '''
    月・火・水・木・金           188
    希望なし                 64
    土・日・祝日               48
    月・火・水・木・金・土・日・祝日      7
    '''
    booking_df["受診会場"].value_counts()
    '''
    加須保健センター・パストラルかぞ・花崎コミュニティセンター                                    140
    大利根健康福祉センター・アスターホール                                               49
    騎西健康福祉センター                                                        36
    北川辺健康福祉センター                                                       31
    加須保健センター・パストラルかぞ・花崎コミュニティセンター・騎西健康福祉センター                          22
    希望なし                                                              12
    加須保健センター・パストラルかぞ・花崎コミュニティセンター・北川辺健康福祉センター・大利根健康福祉センター・アスターホール      5
    加須保健センター・パストラルかぞ・花崎コミュニティセンター・大利根健康福祉センター・アスターホール                  5
    加須保健センター・パストラルかぞ・花崎コミュニティセンター・北川辺健康福祉センター                          3
    北川辺健康福祉センター・大利根健康福祉センター・アスターホール                                    2
    パストラルかぞ                                                            1
    加須保健センター・パストラルかぞ・花崎コミュニティセンター・騎西健康福祉センター・大利根健康福祉センター・アスターホール       1
    '''
