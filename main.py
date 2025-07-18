"""
pm10_forecaster.py

This script reads a JSON input file containing a list of cases and, for each case,
generates a synthetic hourly PM10 forecast at the case’s target location.

Input JSON schema:
{
  "cases": [
    {
      "case_id":        string,
      "stations": [     # list of station objects
        {
          "station_code": string,
          "longitude":    float,
          "latitude":     float,
          "history": [    # list of hourly observations
            {
              "timestamp": str (ISO8601, e.g. "2019-01-01T00:00:00"),
              "pm10":       float
            },
            ...
          ]
        },
        ...
      ],
      "target": {
        "longitude":               float,
        "latitude":                float,
        "prediction_start_time":   str (ISO8601)
      },
      "weather": [ ... ]  # optional array of METAR‐style records
    },
    ...
  ]
}

Usage:
    python pm10_forecaster.py --data-file data.json [--landuse-pbf landuse.pbf] --output-file output.json
"""

import argparse  # For parsing command-line arguments
import json      # For reading and writing JSON files
import random    # For generating random numbers (placeholder for real predictions)
from datetime import datetime, timedelta  # For handling dates and times
import osmium    # For reading OpenStreetMap .pbf files


class LanduseHandler(osmium.SimpleHandler):
    """
    Osmium handler to collect landuse ways and relations from a .pbf file.
    Each object with a 'landuse' tag is stored in a list.
    """
    def __init__(self):
        super().__init__()
        self.landuse_ways = []
        self.landuse_relations = []

    def way(self, w):
        # Called for each way in the .pbf; store if it has a 'landuse' tag
        if 'landuse' in w.tags:
            self.landuse_ways.append({
                "type": "way",
                "id": w.id,
                "landuse": w.tags['landuse'],
                "tags": dict(w.tags),
                # We only store node IDs (refs) here; lat/lon can be resolved later if needed
                "node_refs": [node.ref for node in w.nodes]
            })

    def relation(self, r):
        # Called for each relation in the .pbf; store if it has a 'landuse' tag
        if 'landuse' in r.tags:
            self.landuse_relations.append({
                "type": "relation",
                "id": r.id,
                "landuse": r.tags['landuse'],
                "tags": dict(r.tags),
                # Store member references; for further spatial analysis if needed
                "members": [(m.ref, m.role, m.type) for m in r.members]
            })


def predict_pm10(base_time, history, landuse_data, hours=24):
    """
    Placeholder function to generate a PM10 forecast for a target location.
    - base_time: datetime from which to start predictions
    - history: list of historical PM10 records (unused in this placeholder)
    - landuse_data: optional landuse info (unused in this placeholder)
    - hours: number of hourly predictions to generate
    Returns a list of dictionaries with 'timestamp' and 'pm10_pred'.
    """
    forecast_list = []
    for h in range(hours):
        ts = (base_time + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%MZ")
        # Generate a random PM10 value between 0 and 100 (put your logic here)
        # TODO: replace this random placeholder with your PM10 prediction model
        pm10_pred = round(random.uniform(0, 100), 1)
        forecast_list.append({
            "timestamp": ts,
            "pm10_pred": pm10_pred
        })
    return forecast_list


def generate_output(data, landuse_data=None, forecast_hours=24):
    """
    Generates synthetic PM10 forecasts for each case’s target location.
    - Uses 'prediction_start_time' from each case's target as the base timestamp.
    - Raises an exception if 'prediction_start_time', 'longitude', or 'latitude' are missing/invalid.
    - Optionally prints info about loaded landuse objects.
    - Calls predict_pm10() to obtain the hourly forecast list.
    """
    predictions = []

    if landuse_data:
        total = len(landuse_data["ways"]) + len(landuse_data["relations"])
        print(f"[INFO] Landuse objects loaded: {total}")

    for case in data["cases"]:
        case_id = case["case_id"]
        target = case.get("target")
        if not target or "prediction_start_time" not in target:
            raise ValueError(f"Case '{case_id}' is missing 'prediction_start_time' in target.")

        # Parse 'prediction_start_time' into a datetime object; raise on parse failure
        try:
            base_forecast_start = datetime.fromisoformat(target["prediction_start_time"])
        except Exception as e:
            raise ValueError(f"Invalid prediction_start_time for case '{case_id}': {e}")

        # Ensure both longitude and latitude are present
        longitude = target.get("longitude")
        latitude = target.get("latitude")
        if longitude is None or latitude is None:
            raise ValueError(f"Case '{case_id}' target must include both 'longitude' and 'latitude'.")

        stations = case.get("stations", [])
        print(f"[DEBUG] Generating for case: {case_id}, "
              f"target: ({latitude}, {longitude}), "
              f"start: {base_forecast_start.isoformat()}")
        print(f"[DEBUG] Available stations: {len(stations)}")

        # (Optional) Gather history from all stations for potential future use
        all_history = []
        for station in stations:
            station_code = station["station_code"]
            history = station.get("history", [])
            all_history.extend(history)
            print(f"  [INFO] Station {station_code}: {len(history)} history points")

        # Call the separate prediction function
        forecast_list = predict_pm10(
            base_time=base_forecast_start,
            history=all_history,
            landuse_data=landuse_data,
            hours=forecast_hours
        )

        predictions.append({
            "case_id": case_id,
            "forecast": forecast_list
        })

    return {"predictions": predictions}


def main():
    parser = argparse.ArgumentParser(description="Generate random PM10 forecasts.")
    parser.add_argument("--data-file", required=True, help="Path to input data.json")
    parser.add_argument("--landuse-pbf", required=False, help="Path to landuse.pbf")
    parser.add_argument("--output-file", required=True, help="Path to write output.json")
    args = parser.parse_args()

    # Read the input JSON file containing cases, stations, and target definitions
    with open(args.data_file, "r") as f:
        data = json.load(f)

    landuse_data = None
    if args.landuse_pbf:
        print(f"Reading landuse data from: {args.landuse_pbf}")
        handler = LanduseHandler()
        handler.apply_file(args.landuse_pbf)
        print(f"Found {len(handler.landuse_ways)} landuse ways.")
        print(f"Found {len(handler.landuse_relations)} landuse relations.")

        landuse_data = {
            "ways": handler.landuse_ways,
            "relations": handler.landuse_relations
        }

    # Generate synthetic forecasts (random) for each case’s target
    output = generate_output(data, landuse_data=landuse_data)

    # Write the generated forecasts to the specified output JSON file
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Read input from: {args.data_file}")
    if args.landuse_pbf:
        print(f"Land use PBF provided at: {args.landuse_pbf}")
    print(f"Wrote forecasts to: {args.output_file}")


if __name__ == "__main__":
    main()
