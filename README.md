# SDK_SimulationANDDDoS_detection

## How to collect normal traffic data
1. set up mininet and set h1 the host.
2. figure out ethernet address of h1 and h2 by calling `h1 ifconfig`
3. run h2 bash host_run_normal_traffic.sh. Can specify TCP or UDP. If it's ICMP, simply h1 ping h2
4. run normal_data_collect_script.sh on the background and it collects flow data every 30 seconds