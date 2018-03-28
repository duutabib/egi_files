Directory=`/0/abib`
# -1 = highest priority
# -1023 = lowest priority
Priority="-100"
 
# Available Queues
QueueList="/0/queue_long_64gb.txt"
#QueueList="/0/queue_long.txt"
#QueueList="/0/queue_short_64gb.txt"
#QueueList="/0/queue_short.txt"
  
AllCluster=`cat $QueueList`
  
  echo "We will use: $AllCluster"
  echo ""
  for ((i=1; i < 10; i++ ))
  do
   echo "Start-Job with ID $i"
#   echo 'cd $Directory ; pwd' | sh
   echo 'cd $Directory ; /0/abib/python3/bin/python3  /0/abib/googlenet_script.py ' | qsub -q $AllCluster -p $Priority
 
  done

