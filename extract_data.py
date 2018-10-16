import json
import os

with open('data', 'w+') as g:
	for filename in os.listdir('./json_data'):
		with open('./json_data' + '/' + filename, 'r') as f:
			d = json.load(f)
			g.write(d["Claim"] + '\t' + d["Credibility"] + '\n')



