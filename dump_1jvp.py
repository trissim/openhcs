import json

# Load the benchmark results for 1jvp
result_path = "benchmark_results_debug_1jvp_perfreset1/pdb_redocking_20260328_232311.json"
try:
    with open(result_path, "r") as f:
        data = json.load(f)
except Exception as e:
    print(f"Failed to load {result_path}: {e}")
    sys.exit(1)

print(f"Keys in result: {data.keys()}")
if 'dq_dock' in data and 'results' in data['dq_dock']:
    for res in data['dq_dock']['results']:
        if res['metrics']['pdb_id'] == '1jvp':
            print("Found 1jvp results")
            for k, v in res.get('metadata', {}).items():
                if 'score' in k.lower() or 'energy' in k.lower():
                    print(f"  {k}: {v}")
            if 'pose_rmsd_sorted' in res['metrics']:
                poses = res['metrics']['pose_rmsd_sorted']
                print(f"Total returned poses in json: {len(poses)}")
                for i, p in enumerate(poses[:5]):  # print top 5 poses
                    print(f"Rank {i}:")
                    print(f"  rmsd: {p.get('rmsd')}")
                    print(f"  total_score: {p.get('total_score', p.get('score'))}")
                    print(f"  base_score: {p.get('base_score')}")
                    print(f"  rich_score: {p.get('rich_score')}")
                    print(f"  error_bound: {p.get('error_bound')}")
                    print(f"  metadata: {p.get('metadata')}")
                    print("---")
            break
