import glob
path = './output/PPO_v0_lr*.txt'
for filename in glob.glob(path):
    text = open(filename, 'r').read()
    score_float = float(text.split()[3])
    score = '%.2f' % score_float
    new_name = filename[:16] + f's{score}_obsGreedy_' + filename[16:]
    with open(new_name, 'w') as output:
        output.write(text)