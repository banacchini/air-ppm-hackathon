# 💨 AIR PPM Hackathon Example Application 

This is an example application developed for the AIR PPM Hackathon. It generates synthetic PM10 air pollution forecasts based on provided input data, and can optionally use OpenStreetMap .pbf spatial data (e.g., land use) as additional context.

---

## 📦 Build the Docker Image

Make sure you are in the directory containing the `Dockerfile`, then run:

```bash
docker build -t air-pm10-forecast .
```

This command builds a Docker image named `air-pm10-forecast`.

---

## 🚀 Run the Application

To run the forecast generator, use the following command:

```bash
docker run --rm \
  -v C:/data/data.json:/data/data.json \
  -v C:/data/SpatialData.osm.pbf:/data/landuse.pbf \
  -v C:/data/output.json:/data/output.json \
  air-pm10-forecast \
  --data-file /data/data.json \
  --landuse-pbf /data/landuse.pbf \
  --output-file /data/output.json
```

### 📂 Explanation:

* `--rm`: Automatically removes the container after execution.
* `-v <local>:<container>`: Mounts local files into the container.
* `/data/data.json`: Input data file with case and station information (including historical PM10 data).
* `/data/landuse.pbf`: (Optional) Spatial land use data in OSM `.pbf` format.
* `/data/output.json`: File where forecast results will be saved.
* `air-pm10-forecast`: The Docker image name.
* The final arguments (`--data-file`, etc.) are passed to the application running inside the container.

---

## 📘 Input Format

* `data.json` must include an array of cases with station info, PM10 history and weather history.
* `landuse.pbf` is expected to be a valid OSM `.pbf` file.

---

## 🧪 Output

The application generates `output.json` containing PM10 forecasts for each station in each case.

Each forecast includes:

* 24 hourly PM10 predictions
* Timestamps in ISO 8601 format

---

## 🔧 Notes

* This is a template project: values are randomly generated but all relevant data (history, landuse) is available inside the code for future enhancements.
* You can easily extend the logic in `generate_output()` to use historical or spatial context for smarter predictions.

