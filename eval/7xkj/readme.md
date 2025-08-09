# Usage Instructions

1. First, generate ligands and obtain the JSON file containing some metrics:
    ```bash
    python sample_for_pocket.py /home/xulong/AI/Codes/MSCoD/MSCoD/7xkj_gdp/7xkj_A_rec.pdb /home/xulong/AI/Codes/MSCoD/MSCoD/7xkj_gdp/7xkj_A_rec_7xkj_lig_gdp.sdf

    python sample_for_pocket.py /home/xulong/AI/Codes/MSCoD/MSCoD/7xkj_6ic/7xkj_A_rec.pdb /home/xulong/AI/Codes/MSCoD/MSCoD/7xkj_6ic/7xkj_A_rec_7xkj_lig_6ic.sdf
    ```

2. Then, run the following command to visualize the results:
    ```bash
    python view.py
    ```