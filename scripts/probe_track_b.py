import json, sys, warnings
warnings.filterwarnings("ignore")
from infodiff.experiments.track_b import train_one
ds = sys.argv[1]
for spec in sys.argv[2].split(";"):
    kw = json.loads(spec)
    r = train_one(ds, 0, **kw)
    print(json.dumps(r), flush=True)
