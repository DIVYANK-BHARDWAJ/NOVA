import urllib.request
import json

with open('.env') as f:
    env = dict(l.strip().split('=', 1) for l in f if '=' in l and not l.startswith('#'))
token = env['BOT_ACCESS_TOKEN']
base = env['GATEWAY_URL']

for t in ['HAVN', 'PLNX', 'ROUT', 'SONA']:
    url = f"{base}/api/v1/stocks/history?ticker={t}&start_date=2010-01-01&end_date=2026-04-14"
    req = urllib.request.Request(url)
    req.add_header('Authorization', f"Bearer {token}")
    req.add_header('Cookie', f"access_token={token}")
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            first_p = data.get('prices', [{}])[0].get('date')
            print(f"{t} SUCCEEDED: count = {data.get('total_records')}, first = {first_p}")
    except urllib.error.HTTPError as e:
        print(f"{t} ERROR {e.code}: {e.read().decode()}")

