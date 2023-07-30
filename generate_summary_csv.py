import glob
import pandas as pd
path = './output/PPO_v0_*.txt'
scores = []
obs = []
lrs = []
lambdas = []
gammas = []
epsilons = []

for filename in glob.glob(path):
    split_name = filename.split('_')
    scores.append(float(split_name[2].replace('s', '')))
    obs.append(split_name[3][3:])
    lrs.append(float(split_name[4].replace('lr', '')))
    lambdas.append(float(split_name[5].replace('lambda', '').replace('lb', '')))
    gammas.append(float(split_name[6].replace('gamma', '').replace('g', '')))
    epsilons.append(float(split_name[7].replace('epsilon', '').replace('e', '').replace('.txt', '')))

dict = {
    'Score': scores,
    'Observation': obs,
    'Learning Rate': lrs,
    'Lambda': lambdas,
    'Gamma': gammas,
    'Epsilon': epsilons
}

df = pd.DataFrame(dict)

writer = pd.ExcelWriter('./output/.summary.xlsx', engine = 'xlsxwriter')
df.to_excel(writer, index=False, sheet_name='Sheet1')
workbook = writer.book
worksheet = writer.sheets['Sheet1']
format1 = workbook.add_format({'num_format': '0.00'})
worksheet.set_column('C:C', None, format1)  # Adds formatting to column C
writer.save()