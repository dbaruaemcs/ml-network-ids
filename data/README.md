# Dataset

This project uses the CIC-IDS2017 dataset for network intrusion detection experiments.

The original dataset contains network flow records generated from benign traffic and multiple attack scenarios, including:

* DDoS
* DoS
* PortScan
* Bot
* FTP-Patator
* SSH-Patator
* Web Attacks
* Infiltration
* Heartbleed

The dataset is not included in this repository because of its size.

## Dataset source

CIC-IDS2017 was developed by the Canadian Institute for Cybersecurity.

Download the original dataset from the official CIC-IDS2017 project page and place the CSV files inside this `data/` directory.

## Expected files

The experiments use the following CIC-IDS2017 CSV files:

* Monday-WorkingHours.pcap_ISCX.csv
* Tuesday-WorkingHours.pcap_ISCX.csv
* Wednesday-workingHours.pcap_ISCX.csv
* Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
* Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
* Friday-WorkingHours-Morning.pcap_ISCX.csv
* Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
* Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

After downloading the dataset, place these files in:

```text
ml-network-ids/data/
```

The preprocessing and experiment scripts can then be executed from the project root.
