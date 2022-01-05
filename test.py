import pandas as pd

atena = pd.read_csv('s3://da-cleaning-lgdata-v1/2021/福岡県小郡市/02.mailing/letter/02.second/04.atena/宛名データ（R3）.csv')
mokushi = pd.read_csv('s3://da-cleaning-lgdata-v1/2021/福岡県小郡市/02.mailing/letter/02.second/99.for_work/clean_id/output/要目視確認_カナ氏名突合不可_小郡市除外者データ（12.13現在）.csv')

for name in mokushi['氏名']:
    split_name = name.split('\u3000')
    print(split_name[0], split_name[1])
    atena.loc[atena['氏名'].str.contains(split_name[0]) & atena['氏名'].str.contains(split_name[1]), '氏名一致'] = 1

check = atena[atena['氏名一致'] == 1]
