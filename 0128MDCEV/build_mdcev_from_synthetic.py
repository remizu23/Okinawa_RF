import csv
from collections import defaultdict


def build_mdcev_from_synthetic(
    input_path: str,
    output_path: str,
) -> None:
    counts = defaultdict(lambda: defaultdict(int))
    link_ids = set()
    trip_counts = defaultdict(int)

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trip_id = int(row["trip_id"])
            link_id = int(row["link_id"])
            counts[trip_id][link_id] += 1
            link_ids.add(link_id)
            trip_counts[trip_id] += 1

    link_ids_sorted = sorted(link_ids)
    link_id_to_col = {link_id: idx for idx, link_id in enumerate(link_ids_sorted)}
    link_columns = [f"link{idx + 1}" for idx in range(len(link_ids_sorted))]
    max_transitions = max(trip_counts.values()) if trip_counts else 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + link_columns + ["end"])
        for trip_id in sorted(counts):
            row = [0] * len(link_columns)
            for link_id, count in counts[trip_id].items():
                row[link_id_to_col[link_id]] = count
            end_count = max_transitions - trip_counts[trip_id]
            writer.writerow([trip_id] + row + [end_count])


if __name__ == "__main__":
    build_mdcev_from_synthetic(
        input_path="/Users/masudasatoki/Downloads/okinawa_routechoice/data/input/synthetic_data.csv",
        output_path="/Users/masudasatoki/Downloads/okinawa_routechoice/data/input/mdcev_from_synthetic.csv",
    )
