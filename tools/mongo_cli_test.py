
#!/usr/bin/env python3
import argparse, json
from dotenv import load_dotenv
from iowrapper.mongo_shell import MongoShell

def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Quick Mongo CLI (dotenv-aware)")
    ap.add_argument("--uri", default=None, help="Override MONGODB_URI")
    ap.add_argument("--db", required=True, help="Database name")
    ap.add_argument("--coll", required=True, help="Collection name")
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--interval", default=None)
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    mongo = MongoShell(uri=args.uri, db=args.db)
    c = mongo.get_collection(args.coll, args.db)

    q = {}
    if args.symbol: q["symbol"] = args.symbol
    if args.interval: q["interval"] = args.interval

    docs = list(c.find(q, projection={"_id":0}).sort("endtime", 1).limit(args.limit))
    print(json.dumps(docs, indent=2))

if __name__ == "__main__":
    main()
