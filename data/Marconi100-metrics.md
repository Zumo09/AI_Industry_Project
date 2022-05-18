## Table of contents

[[_TOC_]]

## Introduction
This is a brief description of the metrics collected by ExaMon from the Marconi100 cluster.

## IPMI metrics
The following table describes the metrics collected by the ipmi_pub plugin


| Metric Name      | Description                                                                 | Unit |
|------------------|-----------------------------------------------------------------------------|------|
| pX_coreY_temp    | Temperature of core n. Y in the   CPU socket n. X.  X=0..1, Y=0..23         | °C   |
| dimmX_temp       | Temperature of DIMM module n. X. X=0..15                                    | °C   |
| gpuX_core_temp   | Temperature of the core for the GPU id X. X=0,1,3,4                         | °C   |
| gpuX_mem_temp    | Temperature of the memory for the GPU id    X. X=0,1,3,4                    | °C   |
| fanX_Y           | Speed of the Fan Y in module X.    X=0..3,  Y=0,1                           | RPM  |
| pX_vdd_temp      | Temperature of the voltage regulator for the CPU socket n. X.  X=0..1       | °C   |
| fan_disk_power   | Power consumption of the disk fan                                           | W    |
| pX_io_power      | Power consumption for the I/O subsystem for the CPU socket n. X.  X=0..1    | W    |
| pX_mem_power     | Power consumption for the memory subsystem for the CPU socket n. X.  X=0..1 | W    |
| pX_power         | Power consumption for the CPU socket n. X.  X=0..1                          | W    |
| psX_input_power  | Power consumption at the input of power supply n. X.  X=0..1                | W    |
| total_power      | Total node power consumption                                                | W    |
| psX_input_voltag | Voltage at the input of power supply n. X.  X=0..1                          | V    |
| psX_output_volta | Voltage at the output of power supply n. X.  X=0..1                         | V    |
| psX_output_curre | Current at the output of power supply n. X.  X=0..1                         | A    |
| pcie             | Temperature at the PCIExpress slots                                         | °C   |
| ambient          | Temperature at the node inlet                                               | °C   |

## Ganglia metrics
The following table describes the metrics collected by the ganglia_pub plugin. The data are extracted from a Ganglia (http://ganglia.sourceforge.net/) instance that CINECA runs on Marconi100 and has granted us access to. 

**PLEASE NOTE** 

- The metrics described in this table are only a reference. The actual metrics on Marconi100 may be different.
- The Ganglia instance has frequent failures so data are made available in ExaMon according to a best-effort policy.


| Metric name                      | Type      | Unit        | Description                                                                                                                                                                                                                          |
|----------------------------------|-----------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| gexec                            | core      |             | gexec available                                                                                                                                                                                                                      |
| cpu_aidle                        | cpu       | %           | Percent of time since boot idle   CPU                                                                                                                                                                                                |
| cpu_idle                         | cpu       | %           | Percentage of time that the CPU or   CPUs were idle and the system did not have an outstanding disk I/O request                                                                                                                      |
| cpu_nice                         | cpu       | %           | Percentage of CPU utilization that   occurred while executing at the user level with nice priority                                                                                                                                   |
| cpu_speed                        | cpu       | MHz         | CPU Speed in terms of MHz                                                                                                                                                                                                            |
| cpu_steal                        | cpu       | %           | cpu_steal                                                                                                                                                                                                                            |
| cpu_system                       | cpu       | %           | Percentage of CPU utilization that   occurred while executing at the system level                                                                                                                                                    |
| cpu_user                         | cpu       | %           | Percentage of CPU utilization that   occurred while executing at the user level                                                                                                                                                      |
| cpu_wio                          | cpu       | %           | Percentage of time that the CPU or   CPUs were idle during which the system had an outstanding disk I/O request                                                                                                                      |
| cpu_num                          |           |             |                                                                                                                                                                                                                                      |
| disk_free                        | disk      | GB          | Total free disk space                                                                                                                                                                                                                |
| disk_total                       | disk      | GB          | Total available disk space                                                                                                                                                                                                           |
| part_max_used                    | disk      | %           | Maximum percent used for all   partitions                                                                                                                                                                                            |
| load_fifteen                     | load      |             | Fifteen minute load average                                                                                                                                                                                                          |
| load_five                        | load      |             | Five minute load average                                                                                                                                                                                                             |
| load_one                         | load      |             | One minute load average                                                                                                                                                                                                              |
| mem_buffers                      | memory    | KB          | Amount of buffered memory                                                                                                                                                                                                            |
| mem_cached                       | memory    | KB          | Amount of cached memory                                                                                                                                                                                                              |
| mem_free                         | memory    | KB          | Amount of available memory                                                                                                                                                                                                           |
| mem_shared                       | memory    | KB          | Amount of shared memory                                                                                                                                                                                                              |
| mem_total                        | memory    | KB          | Total amount of memory displayed   in KBs                                                                                                                                                                                            |
| swap_free                        | memory    | KB          | Amount of available swap memory                                                                                                                                                                                                      |
| swap_total                       | memory    | KB          | Total amount of swap space   displayed in KBs                                                                                                                                                                                        |
| bytes_in                         | network   | bytes/sec   | Number of bytes in per second                                                                                                                                                                                                        |
| bytes_out                        | network   | bytes/sec   | Number of bytes out per second                                                                                                                                                                                                       |
| pkts_in                          | network   | packets/sec | Packets in per second                                                                                                                                                                                                                |
| pkts_out                         | network   | packets/sec | Packets out per second                                                                                                                                                                                                               |
| proc_run                         | process   |             | Total number of running processes                                                                                                                                                                                                    |
| proc_total                       | process   |             | Total number of processes                                                                                                                                                                                                            |
| boottime                         | system    | s           | The last time that the system was   started                                                                                                                                                                                          |
| machine_type                     | system    |             | System architecture                                                                                                                                                                                                                  |
| os_name                          | system    |             | Operating system name                                                                                                                                                                                                                |
| os_release                       | system    |             | Operating system release date                                                                                                                                                                                                        |
| cpu_ctxt                         | cpu       | ctxs/sec    | Context Switches                                                                                                                                                                                                                     |
| cpu_intr                         | cpu       | %           | cpu_intr                                                                                                                                                                                                                             |
| cpu_sintr                        | cpu       | %           | cpu_sintr                                                                                                                                                                                                                            |
| multicpu_idle0                   | cpu       | %           | Percentage of CPU utilization that   occurred while executing at the idle level                                                                                                                                                      |
| procs_blocked                    | cpu       | processes   | Processes blocked                                                                                                                                                                                                                    |
| procs_created                    | cpu       | proc/sec    | Number of processes and threads   created                                                                                                                                                                                            |
| disk_free_absolute_developers    | disk      | GB          | Disk space available (GB) on   /developers                                                                                                                                                                                           |
| disk_free_percent_developers     | disk      | %           | Disk space available (%) on   /developers                                                                                                                                                                                            |
| diskstat_sda_io_time             | diskstat  | s           | The time in seconds spent in I/O   operations                                                                                                                                                                                        |
| diskstat_sda_percent_io_time     | diskstat  | percent     | The percent of disk time spent on   I/O operations                                                                                                                                                                                   |
| diskstat_sda_read_bytes_per_sec  | diskstat  | bytes/sec   | The number of bytes read per   second                                                                                                                                                                                                |
| diskstat_sda_reads_merged        | diskstat  | reads       | The number of reads merged. Reads   which are adjacent to each other may be merged for efficiency. Multiple reads   may become one before it is handed to the disk, and it will be counted (and   queued) as only one I/O.           |
| diskstat_sda_reads               | diskstat  | reads       | The number of reads completed                                                                                                                                                                                                        |
| diskstat_sda_read_time           | diskstat  | s           | The time in seconds spent reading                                                                                                                                                                                                    |
| diskstat_sda_weighted_io_time    | diskstat  | s           | The weighted time in seconds spend   in I/O operations. This measures each I/O start, I/O completion, I/O merge,   or read of these stats by the number of I/O operations in progress times the   number of seconds spent doing I/O. |
| diskstat_sda_write_bytes_per_sec | diskstat  | bytes/sec   | The number of bytes written per   second                                                                                                                                                                                             |
| diskstat_sda_writes_merged       | diskstat  | writes      | The number of writes merged.   Writes which are adjacent to each other may be merged for efficiency.   Multiple writes may become one before it is handed to the disk, and it will   be counted (and queued) as only one I/O.        |
| diskstat_sda_writes              | diskstat  | writes      | The number of writes completed                                                                                                                                                                                                       |
| diskstat_sda_write_time          | diskstat  | s           | The time in seconds spent writing                                                                                                                                                                                                    |
| ipmi_ambient_temp                | ipmi      | C           | IPMI data                                                                                                                                                                                                                            |
| ipmi_avg_power                   | ipmi      | Watts       | IPMI data                                                                                                                                                                                                                            |
| ipmi_cpu1_temp                   | ipmi      | C           | IPMI data                                                                                                                                                                                                                            |
| ipmi_cpu2_temp                   | ipmi      | C           | IPMI data                                                                                                                                                                                                                            |
| ipmi_gpu_outlet_temp             | ipmi      | C           | IPMI data                                                                                                                                                                                                                            |
| ipmi_hdd_inlet_temp              | ipmi      | C           | IPMI data                                                                                                                                                                                                                            |
| ipmi_pch_temp                    | ipmi      | C           | IPMI data                                                                                                                                                                                                                            |
| ipmi_pci_riser_1_temp            | ipmi      | C           | IPMI data                                                                                                                                                                                                                            |
| ipmi_pci_riser_2_temp            | ipmi      | C           | IPMI data                                                                                                                                                                                                                            |
| ipmi_pib_ambient_temp            | ipmi      | C           | IPMI data                                                                                                                                                                                                                            |
| mem_anonpages                    | memory    | Bytes       | AnonPages                                                                                                                                                                                                                            |
| mem_dirty                        | memory    | Bytes       | The total amount of memory waiting   to be written back to the disk.                                                                                                                                                                 |
| mem_hardware_corrupted           | memory    | Bytes       | HardwareCorrupted                                                                                                                                                                                                                    |
| mem_mapped                       | memory    | Bytes       | Mapped                                                                                                                                                                                                                               |
| mem_writeback                    | memory    | Bytes       | The total amount of memory   actively being written back to the disk.                                                                                                                                                                |
| vm_pgmajfault                    | memory_vm | ops/s       | pgmajfault                                                                                                                                                                                                                           |
| vm_pgpgin                        | memory_vm | ops/s       | pgpgin                                                                                                                                                                                                                               |
| vm_pgpgout                       | memory_vm | ops/s       | pgpgout                                                                                                                                                                                                                              |
| vm_vmeff                         | memory_vm | pct         | VM efficiency                                                                                                                                                                                                                        |
| rx_bytes_eth0                    | network   | bytes/sec   | received bytes per sec                                                                                                                                                                                                               |
| rx_drops_eth0                    | network   | pkts/sec    | receive packets dropped per sec                                                                                                                                                                                                      |
| rx_errs_eth0                     | network   | pkts/sec    | received error packets per sec                                                                                                                                                                                                       |
| rx_pkts_eth0                     | network   | pkts/sec    | received packets per sec                                                                                                                                                                                                             |
| tx_bytes_eth0                    | network   | bytes/sec   | transmitted bytes per sec                                                                                                                                                                                                            |
| tx_drops_eth0                    | network   | pkts/sec    | transmitted dropped packets per   sec                                                                                                                                                                                                |
| tx_errs_eth0                     | network   | pkts/sec    | transmitted error packets per sec                                                                                                                                                                                                    |
| tx_pkts_eth0                     | network   | pkts/sec    | transmitted packets per sec                                                                                                                                                                                                          |
| procstat_gmond_cpu               | procstat  | percent     | The total percent CPU utilization                                                                                                                                                                                                    |
| procstat_gmond_mem               | procstat  | B           | The total memory utilization                                                                                                                                                                                                         |
| softirq_blockiopoll              | softirq   | ops/s       | Soft Interrupts                                                                                                                                                                                                                      |
| softirq_block                    | softirq   | ops/s       | Soft Interrupts                                                                                                                                                                                                                      |
| softirq_hi                       | softirq   | ops/s       | Soft Interrupts                                                                                                                                                                                                                      |
| softirq_hrtimer                  | softirq   | ops/s       | Soft Interrupts                                                                                                                                                                                                                      |
| softirq_netrx                    | softirq   | ops/s       | Soft Interrupts                                                                                                                                                                                                                      |
| softirq_nettx                    | softirq   | ops/s       | Soft Interrupts                                                                                                                                                                                                                      |
| softirq_rcu                      | softirq   | ops/s       | Soft Interrupts                                                                                                                                                                                                                      |
| softirq_sched                    | softirq   | ops/s       | Soft Interrupts                                                                                                                                                                                                                      |
| softirq_tasklet                  | softirq   | ops/s       | Soft Interrupts                                                                                                                                                                                                                      |
| softirq_timer                    | softirq   | ops/s       | Soft Interrupts                                                                                                                                                                                                                      |
| entropy_avail                    | ssl       | bits        | Entropy Available                                                                                                                                                                                                                    |
| tcpext_listendrops               | tcpext    | count/s     | listendrops                                                                                                                                                                                                                          |
| tcpext_tcploss_percentage        | tcpext    | pct         | TCP percentage loss, tcploss /   insegs + outsegs                                                                                                                                                                                    |
| tcp_attemptfails                 | tcp       | count/s     | attemptfails                                                                                                                                                                                                                         |
| tcp_insegs                       | tcp       | count/s     | insegs                                                                                                                                                                                                                               |
| tcp_outsegs                      | tcp       | count/s     | outsegs                                                                                                                                                                                                                              |
| tcp_retrans_percentage           | tcp       | pct         | TCP retrans percentage,   retranssegs / insegs + outsegs                                                                                                                                                                             |
| udp_indatagrams                  | udp       | count/s     | indatagrams                                                                                                                                                                                                                          |
| udp_inerrors                     | udp       | count/s     | inerrors                                                                                                                                                                                                                             |
| udp_outdatagrams                 | udp       | count/s     | outdatagrams                                                                                                                                                                                                                         |
| multicpu_idle16                  | cpu       | %           | Percentage of CPU utilization that   occurred while executing at the idle level                                                                                                                                                      |
| multicpu_steal16                 | cpu       | %           | Percentage of CPU preempted by the   hypervisor                                                                                                                                                                                      |
| multicpu_system16                | cpu       | %           | Percentage of CPU utilization that   occurred while executing at the system level                                                                                                                                                    |
| multicpu_user16                  | cpu       | %           | Percentage of CPU utilization that   occurred while executing at the user level                                                                                                                                                      |
| multicpu_wio16                   | cpu       | %           | Percentage of CPU utilization that   occurred while executing at the wio level                                                                                                                                                       |
| diskstat_sdb_io_time             | diskstat  | s           | The time in seconds spent in I/O   operations                                                                                                                                                                                        |
| diskstat_sdb_percent_io_time     | diskstat  | percent     | The percent of disk time spent on   I/O operations                                                                                                                                                                                   |
| diskstat_sdb_read_bytes_per_sec  | diskstat  | bytes/sec   | The number of bytes read per   second                                                                                                                                                                                                |
| diskstat_sdb_reads_merged        | diskstat  | reads       | The number of reads merged. Reads   which are adjacent to each other may be merged for efficiency. Multiple reads   may become one before it is handed to the disk, and it will be counted (and   queued) as only one I/O.           |
| diskstat_sdb_reads               | diskstat  | reads       | The number of reads completed                                                                                                                                                                                                        |
| diskstat_sdb_read_time           | diskstat  | s           | The time in seconds spent reading                                                                                                                                                                                                    |
| diskstat_sdb_weighted_io_time    | diskstat  | s           | The weighted time in seconds spend   in I/O operations. This measures each I/O start, I/O completion, I/O merge,   or read of these stats by the number of I/O operations in progress times the   number of seconds spent doing I/O. |
| diskstat_sdb_write_bytes_per_sec | diskstat  | bytes/sec   | The number of bytes written per   second                                                                                                                                                                                             |
| diskstat_sdb_writes_merged       | diskstat  | writes      | The number of writes merged.   Writes which are adjacent to each other may be merged for efficiency.   Multiple writes may become one before it is handed to the disk, and it will   be counted (and queued) as only one I/O.        |
| diskstat_sdb_writes              | diskstat  | writes      | The number of writes completed                                                                                                                                                                                                       |
| diskstat_sdb_write_time          | diskstat  | s           | The time in seconds spent writing                                                                                                                                                                                                    |
| GpuX_dec_utilization        | gpu | %       |X=0,..,3   | 
| GpuX_enc_utilization        | gpu | %       |   |
| GpuX_enforced_power_limit   | gpu | Watts   |   |
| GpuX_gpu_temp               | gpu | Celsius |   |
| GpuX_low_util_violation     | gpu |         |   |
| GpuX_mem_copy_utilization   | gpu | %       |   |
| GpuX_mem_util_samples       | gpu |         |   |
| GpuX_memory_clock           | gpu | Mhz     |   |
| GpuX_memory_temp            | gpu | Celsius |   |
| GpuX_power_management_limit | gpu | Watts   |   |
| GpuX_power_usage            | gpu | Watts   |   |
| GpuX_pstate                 | gpu |         |   |
| GpuX_reliability_violation  | gpu |         |   |
| GpuX_sm_clock               | gpu | Mhz     |   |

## Nagios metrics
This is a description of the metrics collected by the ExaMon "nagios_pub" plugin. The data reflect those monitored by the Nagios tool (https://www.nagios.org/) that currently runs in the CINECA clusters. Specifically, the plugin interfaces with a Nagios extension developed by CINECA called "Hnagios" (https://prace-ri.eu/wp-content/uploads/Design_Development_and_Improvement_of_Nagios_System_Monitoring_for_Large_Clusters.pdf). Although the monitored services and metrics are similar between all clusters, here we will specifically discuss those of Marconi100.

### Metrics
Use DESCRIBE to find the metrics collected by the nagios_pub plugin for the Marconi100 cluster 


```python
df = sq.DESCRIBE(tag_key = 'plugin', tag_value='nagios_pub') \
    .DESCRIBE(tag_key = 'cluster', tag_value='marconi100') \
    .JOIN(how='inner') \
    .execute()

df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hostscheduleddowtimecomments</td>
    </tr>
    <tr>
      <th>1</th>
      <td>plugin_output</td>
    </tr>
    <tr>
      <th>2</th>
      <td>state</td>
    </tr>
  </tbody>
</table>
</div>



#### hostscheduleddowtimecomments
This metric is obtained from the "Hnagios" output and reports comments made by system administrators about the maintenance status of the specific monitored resource


```python
df = sq.DESCRIBE(metric = 'hostscheduleddowtimecomments').execute()

df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>tag key</th>
      <th>tag values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hostscheduleddowtimecomments</td>
      <td>node</td>
      <td>[ems02, login03, login08, master01, master02, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hostscheduleddowtimecomments</td>
      <td>slot</td>
      <td>[01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 1...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hostscheduleddowtimecomments</td>
      <td>description</td>
      <td>[afs::blocked_conn::status, afs::bosserver::st...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>hostscheduleddowtimecomments</td>
      <td>plugin</td>
      <td>[nagios_pub]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>hostscheduleddowtimecomments</td>
      <td>chnl</td>
      <td>[data]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>hostscheduleddowtimecomments</td>
      <td>host_group</td>
      <td>[compute, compute,cincompute, efgwcompute, efg...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hostscheduleddowtimecomments</td>
      <td>cluster</td>
      <td>[galileo, marconi, marconi100]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>hostscheduleddowtimecomments</td>
      <td>state</td>
      <td>[0, 1, 2, 3]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>hostscheduleddowtimecomments</td>
      <td>nagiosdrained</td>
      <td>[0, 1]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>hostscheduleddowtimecomments</td>
      <td>org</td>
      <td>[cineca]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>hostscheduleddowtimecomments</td>
      <td>state_type</td>
      <td>[0, 1]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>hostscheduleddowtimecomments</td>
      <td>rack</td>
      <td>[205, 206, 207, 208, 209, 210, 211, 212, 213, ...</td>
    </tr>
  </tbody>
</table>
</div>



In the following query, you can have an example of the values contained in this metric (strings)


```python
data = sq.SELECT('node','description','state') \
    .FROM('hostscheduleddowtimecomments') \
    .WHERE(plugin='nagios_pub', cluster='marconi100') \
    .TSTART(1, 'months') \
    .execute()

data.df_table.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>description</th>
      <th>name</th>
      <th>node</th>
      <th>state</th>
      <th>timestamp</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>alive::ping</td>
      <td>hostscheduleddowtimecomments</td>
      <td>ems02</td>
      <td>0</td>
      <td>2021-02-16 15:30:00.097000+01:00</td>
      <td>ST_5-279 test reinstall RHEL8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>alive::ping</td>
      <td>hostscheduleddowtimecomments</td>
      <td>ems02</td>
      <td>0</td>
      <td>2021-02-16 15:45:00.097000+01:00</td>
      <td>ST_5-279 test reinstall RHEL8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>alive::ping</td>
      <td>hostscheduleddowtimecomments</td>
      <td>ems02</td>
      <td>0</td>
      <td>2021-02-17 08:15:00.098000+01:00</td>
      <td>ST_5-279 test reinstall RHEL8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>alive::ping</td>
      <td>hostscheduleddowtimecomments</td>
      <td>ems02</td>
      <td>0</td>
      <td>2021-02-17 08:30:00.097000+01:00</td>
      <td>ST_5-279 test reinstall RHEL8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>alive::ping</td>
      <td>hostscheduleddowtimecomments</td>
      <td>ems02</td>
      <td>0</td>
      <td>2021-02-17 08:45:00.101000+01:00</td>
      <td>ST_5-279 test reinstall RHEL8</td>
    </tr>
  </tbody>
</table>
</div>




```python
list(data.df_table.value.unique())
```




    [u'ST_5-279 test reinstall RHEL8',
     u'Vedi issue SDHPCSY-26329',
     u'+++ MULTIPLE DOWNTIMES +++ Vedi issue SDHPCSY-26329 +++ Vedi issue SDHPCSY-26329',
     u'+++ MULTIPLE DOWNTIMES +++ Vedi issue SDHPCSY-26329 +++ Vedi issue SDHPCSY-26329 +++ ST_5-279 drenato temporanemente fino a quando reinstalliamo in master con rh8',
     u'+++ MULTIPLE DOWNTIMES +++ Vedi issue SDHPCSY-26329 +++ Vedi issue SDHPCSY-26329 +++ Vedi issue SDHPCSY-26329',
     u'SDHPCSY-23582 FP AP in lavorazione per sga3 progetto europeo',
     u'ST_5-279 test reinstall RHEL8 - nodo quorum manager rh7',
     u'CINGOD GPU retired pages found',
     u'SDHPCSY-26449 nodo irraggiungibile da verificare (Andrea Pieretti by portal)',
     u'Da reboottare - SDHPCSY-26428',
     u'SDHPCSY-26410 Fan rotta (Marco Alberoni by portal)',
     u'Nodi lenti - SDHPCSY-26325',
     u'Nodo da verificare - SDHPCSY-26323',
     u'SDHPCSY-26242 nodi con gpu calde da controllare (Andrea Acquaviva by portal)',
     u'SDHPCSY-26391 Manca una IB (Marco Alberoni by portal)',
     u'GPU calda da testare - SDHPCSY-26411',
     u'+++ MULTIPLE DOWNTIMES +++ GPU calda da testare - SDHPCSY-26411 +++ SDHPCSY-26411 GPU calda (Marco Alberoni by portal)',
     u'SDHPCSY-26476 CPU guasta (Marco Alberoni by portal)',
     u'Sembrano lenti - SDHPCSY-26426',
     u'SDHPCSY-26412 CPU guasta (Marco Alberoni by portal)',
     u'SDHPCSY-26442 banco di ram guasto (Andrea Pieretti by portal)',
     u'SDHPCSY-26072 Segnalati problema alla GPU (Giuseppe Palumbo by portal)',
     u'SDHPCSY-26163 CPU guasta (Marco Alberoni by portal)',
     u'SDHPCSY-26074 necessario reboot dopo sostituzione ventola']



#### plugin_output
This metric collects the outbound message from Nagios agents responsible for monitoring services


```python
df = sq.DESCRIBE(metric = 'plugin_output').execute()

df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>tag key</th>
      <th>tag values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>plugin_output</td>
      <td>node</td>
      <td>[ems02, ethcore01-mgt, ethcore02-mgt, gss03, g...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>plugin_output</td>
      <td>slot</td>
      <td>[01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 1...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>plugin_output</td>
      <td>description</td>
      <td>[EFGW_cluster::status::availability, EFGW_clus...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>plugin_output</td>
      <td>plugin</td>
      <td>[nagios_pub]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>plugin_output</td>
      <td>chnl</td>
      <td>[data]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>plugin_output</td>
      <td>host_group</td>
      <td>[compute, compute,cincompute, containers, cumu...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>plugin_output</td>
      <td>cluster</td>
      <td>[galileo, marconi, marconi100]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>plugin_output</td>
      <td>state</td>
      <td>[0, 1, 2, 3]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>plugin_output</td>
      <td>nagiosdrained</td>
      <td>[0, 1]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>plugin_output</td>
      <td>org</td>
      <td>[cineca]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>plugin_output</td>
      <td>state_type</td>
      <td>[0, 1]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>plugin_output</td>
      <td>rack</td>
      <td>[202, 205, 206, 207, 208, 209, 210, 211, 212, ...</td>
    </tr>
  </tbody>
</table>
</div>



In the following query, you can have an example of the values contained in this metric (strings)


```python
data = sq.SELECT('node','description','state') \
    .FROM('plugin_output') \
    .WHERE(plugin='nagios_pub', cluster='marconi100', node='r205n12', state='1,2,3') \
    .TSTART(6, 'months') \
    .execute()

data.df_table.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>description</th>
      <th>name</th>
      <th>node</th>
      <th>state</th>
      <th>timestamp</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>alive::ping</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-09-29 10:15:00.097000+02:00</td>
      <td>CRITICAL - Host Unreachable (10.39.5.12)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>alive::ping</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-10-13 08:45:00.121000+02:00</td>
      <td>CRITICAL - Host Unreachable (10.39.5.12)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>alive::ping</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-10-13 09:00:00.100000+02:00</td>
      <td>CRITICAL - Host Unreachable (10.39.5.12)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>alive::ping</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-10-13 09:15:00.100000+02:00</td>
      <td>CRITICAL - Host Unreachable (10.39.5.12)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>alive::ping</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-10-13 09:30:00.100000+02:00</td>
      <td>CRITICAL - Host Unreachable (10.39.5.12)</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.df_table.drop_duplicates('value')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>description</th>
      <th>name</th>
      <th>node</th>
      <th>state</th>
      <th>timestamp</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>alive::ping</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-09-29 10:15:00.097000+02:00</td>
      <td>CRITICAL - Host Unreachable (10.39.5.12)</td>
    </tr>
    <tr>
      <th>24</th>
      <td>alive::ping</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>3</td>
      <td>2020-09-29 09:45:00.102000+02:00</td>
      <td>(No output on stdout) stderr:</td>
    </tr>
    <tr>
      <th>30</th>
      <td>batchs::client::state</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-09-16 20:45:00.097000+02:00</td>
      <td>ERROR, this mhsc command did not produce a val...</td>
    </tr>
    <tr>
      <th>77</th>
      <td>batchs::client::state</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-09-29 10:30:00.122000+02:00</td>
      <td>DOWN$ matches a critical state</td>
    </tr>
    <tr>
      <th>78</th>
      <td>batchs::client::state</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-09-29 10:45:00.097000+02:00</td>
      <td>DOWN$+DRAIN matches a critical state</td>
    </tr>
    <tr>
      <th>104</th>
      <td>batchs::client::state</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-09-29 17:15:00.100000+02:00</td>
      <td>DOWN+DRAIN matches a critical state</td>
    </tr>
    <tr>
      <th>182</th>
      <td>batchs::client::state</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-10-13 10:00:00.097000+02:00</td>
      <td>ERROR, the command timeouted: /sbin/runuser -c...</td>
    </tr>
    <tr>
      <th>224</th>
      <td>batchs::client</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-09-29 10:15:00.097000+02:00</td>
      <td>Slurmd status: 3, inactive</td>
    </tr>
    <tr>
      <th>236</th>
      <td>filesys::eurofusion::mount</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-09-29 10:15:00.097000+02:00</td>
      <td>C(/m100[umount],/m100_scratch[umount],/m100_wo...</td>
    </tr>
    <tr>
      <th>238</th>
      <td>filesys::eurofusion::mount</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-09-30 14:15:00.100000+02:00</td>
      <td>C(/m100_wai[umount]) W() O(/m100,/m100_scratch...</td>
    </tr>
    <tr>
      <th>240</th>
      <td>filesys::eurofusion::mount</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-10-13 14:45:00.097000+02:00</td>
      <td>C(/m100[umount],/m100_scratch[umount],/m100_wa...</td>
    </tr>
    <tr>
      <th>264</th>
      <td>nvidia::memory::retirement</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-09-29 10:15:00.097000+02:00</td>
      <td>got return value 1 and void stdout from comman...</td>
    </tr>
    <tr>
      <th>268</th>
      <td>ssh::daemon</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-09-29 10:15:00.097000+02:00</td>
      <td>connect to address 10.39.5.12 and port 22: No ...</td>
    </tr>
    <tr>
      <th>292</th>
      <td>ssh::daemon</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-10-13 16:30:00.097000+02:00</td>
      <td>CRITICAL - Socket timeout</td>
    </tr>
    <tr>
      <th>297</th>
      <td>sys::gpfs::status</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-09-29 10:15:00.097000+02:00</td>
      <td>C({mmdiag=11}[/m100]) W() O()</td>
    </tr>
    <tr>
      <th>299</th>
      <td>sys::gpfs::status</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-09-30 14:15:00.100000+02:00</td>
      <td>C({Timeouted[mmgetstate]}[/m100]) W() O()</td>
    </tr>
    <tr>
      <th>300</th>
      <td>sys::gpfs::status</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2020-10-13 14:45:00.097000+02:00</td>
      <td>C({mmgestate=down}[/m100]) W() O()</td>
    </tr>
    <tr>
      <th>311</th>
      <td>sys::rvitals</td>
      <td>plugin_output</td>
      <td>r205n12</td>
      <td>2</td>
      <td>2021-02-17 08:30:00.097000+01:00</td>
      <td>C(rvitals: old data, last update more than 1h ...</td>
    </tr>
  </tbody>
</table>
</div>



#### state
This metric collects the equivalent numerical value of the actual state of the service monitored by Nagios


```python
df = sq.DESCRIBE(metric = 'state').execute()

df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>tag key</th>
      <th>tag values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>state</td>
      <td>node</td>
      <td>[ems02, ethcore01-mgt, ethcore02-mgt, gss03, g...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>state</td>
      <td>slot</td>
      <td>[01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 1...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>state</td>
      <td>description</td>
      <td>[EFGW_cluster::status::availability, EFGW_clus...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>state</td>
      <td>plugin</td>
      <td>[nagios_pub]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>state</td>
      <td>chnl</td>
      <td>[data]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>state</td>
      <td>host_group</td>
      <td>[compute, compute,cincompute, containers, cumu...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>state</td>
      <td>cluster</td>
      <td>[galileo, marconi, marconi100]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>state</td>
      <td>nagiosdrained</td>
      <td>[0, 1]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>state</td>
      <td>org</td>
      <td>[cineca]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>state</td>
      <td>state_type</td>
      <td>[0, 1]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>state</td>
      <td>rack</td>
      <td>[202, 205, 206, 207, 208, 209, 210, 211, 212, ...</td>
    </tr>
  </tbody>
</table>
</div>


In the following query, you can have an example of the values contained in this metric

```python
data = sq.SELECT('node','description','state') \
    .FROM('state') \
    .WHERE(plugin='nagios_pub', cluster='marconi100', node='r205n12') \
    .TSTART(6, 'months') \
    .execute()

data.df_table.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>description</th>
      <th>name</th>
      <th>node</th>
      <th>state</th>
      <th>timestamp</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>alive::ping</td>
      <td>state</td>
      <td>r205n12</td>
      <td></td>
      <td>2020-09-09 17:45:00.097000+02:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>alive::ping</td>
      <td>state</td>
      <td>r205n12</td>
      <td></td>
      <td>2020-09-09 18:00:00.097000+02:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>alive::ping</td>
      <td>state</td>
      <td>r205n12</td>
      <td></td>
      <td>2020-09-09 18:15:00.100000+02:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>alive::ping</td>
      <td>state</td>
      <td>r205n12</td>
      <td></td>
      <td>2020-09-09 18:30:00.101000+02:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>alive::ping</td>
      <td>state</td>
      <td>r205n12</td>
      <td></td>
      <td>2020-09-09 18:45:00.098000+02:00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
list(data.df_table.value.unique())
```




    [0, 3, 2]



### Resources monitored in Marconi100
The name and type of the services/resources monitored by Nagios and corresponding to the metrics just described above are collected in the "description" tag. We use DESCRIBE as follows to get all the services monitored by Nagios for the "Marconi100" cluster:


```python
df_services = sq.DESCRIBE(tag_key='description', where='cluster==marconi100').execute()

df_services
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>tag key</th>
      <th>tag values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>plugin_output</td>
      <td>description</td>
      <td>[alive::ping, backup::local::status, batchs::c...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hostscheduleddowtimecomments</td>
      <td>description</td>
      <td>[alive::ping, backup::local::status, batchs::c...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>state</td>
      <td>description</td>
      <td>[alive::ping, backup::local::status, batchs::c...</td>
    </tr>
  </tbody>
</table>
</div>



The values of the "description" tag are the same for all metrics related to a given cluster. For example, let's see the services related to the "plugin_output" metric, which therefore correspond to the list of all the services monitored by Nagios in the Marconi100 cluster.


```python
df_services[df_services['name'] == 'plugin_output']['tag values'].to_list()[0]
```




    [u'alive::ping',
     u'backup::local::status',
     u'batchs::client',
     u'batchs::client::serverrespond',
     u'batchs::client::state',
     u'batchs::manager::state',
     u'bmc::events',
     u'cluster::status::availability',
     u'cluster::status::criticality',
     u'cluster::status::internal',
     u'container::check::health',
     u'container::check::internal',
     u'container::check::mounts',
     u'dev::raid::status',
     u'dev::swc::confcheck',
     u'dev::swc::confcheckself',
     u'dev::swc::cumulushealth',
     u'dev::swc::cumulussensors',
     u'dev::swc::mlxhealth',
     u'dev::swc::mlxsensors',
     u'file::integrity',
     u'filesys::dres::mount',
     u'filesys::eurofusion::mount',
     u'filesys::local::avail',
     u'filesys::local::mount',
     u'galera::status::Integrity',
     u'galera::status::NodeStatus',
     u'galera::status::ReplicaStatus',
     u'globus::gridftp',
     u'globus::gsissh',
     u'memory::phys::total',
     u'monitoring::health',
     u'net::ib::status',
     u'nfs::rpc::status',
     u'nvidia::configuration',
     u'nvidia::memory::replace',
     u'nvidia::memory::retirement',
     u'service::cert',
     u'service::galera',
     u'service::galera:arbiter',
     u'service::galera:mysql',
     u'ssh::daemon',
     u'sys::corosync::rings',
     u'sys::gpfs::status',
     u'sys::pacemaker::crm',
     u'sys::rvitals']


#### Nagios checks for Marconi100
In the following table is collected a brief description of the services obtained from the query

| Service/resource      | Description                       |
|-----------------------|-----------------------------------|
| alive::ping           | Ping command output               |
| backup::local::status | Backup service                    |
| batchs::…             | Batch scheduler services          |
| bmc::events           | Events from the node BMC          |
| cluster::…            | Cluster availability              |
| container::…          | Status of the container system    |
| dev::…                | Node devices                      |
| file::integrity       | Files integrity                   |
| filesys::….           | Filesystem elements               |
| galera::…             | Status of the database components |
| globus::…             | Status of the FTP system          |
| memory::phys::total   | Physical memory size              |
| monitoring::health    | Monitoring subsystem              |
| net::ib::status       | Infiniband                        |
| nfs::rpc::status      | NFS                               |
| nvidia::…             | GPUs                              |
| service::…            | Misc. services                    |
| ssh::…                | SSH server                        |
| sys::…                | Misc. systems (GPFS,…)            |

### Nagios state encoding 

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Element   </th>
    <th class="tg-0pky">Description</th>
    <th class="tg-0pky"> Encoding  <br>    </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">   <br>state   </td>
    <td class="tg-0pky">   <br>Number   indicating the state of host or service when the event handler was run   </td>
    <td class="tg-0pky">   <br>For   host event handlers: <br>   <br>0 = UP<br>   <br>1 =   DOWN<br>   <br>2 =   UNREACHABLE<br>   <br>For   service event handlers:<br>   <br>0 = OK<br>   <br>1 =   WARNING<br>   <br>2 =   CRITICAL<br>   <br>3 =   UNKNOWN   </td>
  </tr>
  <tr>
    <td class="tg-0pky">   <br>state_type   </td>
    <td class="tg-0pky">   <br>Number   indicating the state type of the host or service when the event handler was   run.   </td>
    <td class="tg-0pky">   <br>0 =   SOFT state<br>   <br>1 =   HARD state   </td>
  </tr>
</tbody>
</table>

## Nvidia metrics
The following table describes the metrics collected by the nvidia_pub plugin.

**PLEASE NOTE**
This plugin has collected data only for a short period (January/February 2020) and is currently not enabled due to CINECA policy



| Metric name                           | Descritpion                                                                                                                                                              | Unit |
|---------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
|     clock.sm                          |     Current frequency   of SM (Streaming Multiprocessor) clock.                                                                                                          | MHz  |
|     clocks.gr                         |     Current frequency   of graphics (shader) clock.                                                                                                                      | MHz  |
|     clocks.mem                        |     Current frequency   of memory clock.                                                                                                                                 | MHz  |
|     clocks_throttle_reasons.active    |     Bitmask of active   clock throttle reasons. See nvml.h for more details                                                                                              |      |
|     power.draw                        |     The last measured   power draw for the entire board, in watts. Only available if power management   is supported. This reading is accurate to within +/- 5 watts.    | W    |
|     temperature.gpu                   |     Core GPU   temperature. in degrees C.                                                                                                                                | °C   |

