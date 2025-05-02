# PASSAT: Physics-assisted and topology-informed Weather Prediction
PASSAT is a novel deep learning model for weather prediction. PASSAT attributes the weather evolution to two key factors: (i) the advection process that can be characterized by the advection equation and the
Navier-Stokes equation; (ii) the Earth-atmosphere interaction that is difficult to both model and calculate. PASSAT also takes the topology of the Earth’s surface into consideration, other than simply treating it as a plane. With these considerations, PASSAT numerically solves the advection equation and the Navier-Stokes equation on the spherical manifold, utilizes a spherical graph neural network to capture the Earth-atmosphere interaction, and generates the initial velocity fields that are critical to solving the advection equation from the same spherical graph neural network.

## Updates
#04/28/2025
:tada: PASSAT has been accepted for IJCAI 2025.

## Usage

### Install

- Create a conda virtual environment and activate it:

```bash
conda create -n PASSAT python=3.8 -y
conda activate PASSAT
```

- Install `CUDA>=11.8` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=2.1.2` and `torchvision>=0.16.2` with `CUDA>=11.8`:

```bash
conda install --user pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

- Install other requirements:

```bash
pip install timm==0.4.12 opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy xarray netcdf4
```

### Data preparation

We use standard 5.625° ERA5 dataset, you can download it from https://github.com/pangeo-data/WeatherBench or run:

```bash
cd data
python -m download
```

- The file structure should look like:
  ```bash
  $ tree data
  ERA5
  ├── 2m_temperature
  │   ├── 2m_temperature_1979_5.625deg.nc
  │   ├── 2m_temperature_1980_5.625deg.nc
  │   └── ...
  ├── 10m_u_component_of_wind
  │   ├── 10m_u_component_of_wind_1979_5.625deg.nc
  │   ├── 10m_u_component_of_wind_1980_5.625deg.nc
  │   └── ...
  │   ...
  ├── geopotential_500
  │   ├── geopotential_500hPa_1979_5.625deg.nc
  │   ├── geopotential_500hPa_1980_5.625deg.nc
  │   └── ...
  ├── temperature_850
  │   ├── temperature_850hPa_1979_5.625deg.nc
  │   ├── temperature_850hPa_1980_5.625deg.nc
  │   └── ...
    ```

After downloading the ERA5 data, we change the file name of geopotential_500 and temperature_850 for convenience as following:

- The file structure should look like:
  ```bash
  $ tree data
  ERA5
  ├── 2m_temperature
  │   ├── 2m_temperature_1979_5.625deg.nc
  │   ├── 2m_temperature_1980_5.625deg.nc
  │   └── ...
  ├── 10m_u_component_of_wind
  │   ├── 10m_u_component_of_wind_1979_5.625deg.nc
  │   ├── 10m_u_component_of_wind_1980_5.625deg.nc
  │   └── ...
  │   ...
  ├── geopotential_500hPa
  │   ├── geopotential_500hPa_1979_5.625deg.nc
  │   ├── geopotential_500hPa_1980_5.625deg.nc
  │   └── ...
  ├── temperature_850hPa
  │   ├── temperature_850hPa_1979_5.625deg.nc
  │   ├── temperature_850hPa_1980_5.625deg.nc
  │   └── ...
    ```

To save memory, we stored the Earth system variables for each hour of ERA5 data and recorded them in a chronological list: 

```bash
python -m preliminary
```

### Evaluation

To evaluate a pre-trained `PASSAT` on ERA5 test set (at 2017, 2018), run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval \
--cfg <config-file> --pretrained <checkpoint> 
```

For example, to evaluate the `PASSAT` with 4 GPUs:

```bash
python -m torch.distributed.run --nproc_per_node 4 --master_port 12345 main.py --eval  \
--cfg configs/PASSAT.yaml --pretrained PASSAT.pth
```

### Training from scratch on ERA5 dataset

To train a `PASSAT` on ERA5 from scratch, execute pre-training and fine-tuning commands in sequence:

Pre-training, or auto-resuming from pre-training, run:
```bash
python -m torch.distributed.run --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py \
--cfg <config-file>
```

**Notes**:
- To save GPU memory, we used mixed-precision training. 
- When GPU memory is not enough, you can try the following suggestions:
    - Use gradient accumulation by adding `--accumulation-steps <steps>`, set appropriate `<steps>` according to your need.

For example, to train `PASSAT` with 4 GPU on a single node for 50 epochs, run:

```bash
python -m torch.distributed.run --nproc_per_node 4 --master_port 12345 main.py \
--cfg configs/PASSAT.yaml
```
