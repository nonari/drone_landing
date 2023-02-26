big=("0043" "0044" "0045" "0046" "0047" "0050" "0053" "0085" "0093" "0097" "0101" "0114" "0118" )
times=(4      1      1      2      1      2      1      1      1      5      1      2      3)

i=0
for vid in ${big[*]}
do
    iters=${times[i]}
    echo $iters
    ini=0
    end=2000
    j=0
    while ((j < iters))
    do
        python3 ~/PycharmProjects/drone_landing/datasets/generate_flow_db.py -vid="$vid" -start="$ini" -end="$end"
        python3 ~/PycharmProjects/segprop2/ruralscapes_demo.py -vid="$vid" -start_from="$ini" -stop_at="$end"
        rm ~/ruralscapes/flow_farneback/*
        ini=$((ini+2000))
        end=$((end+2000))
        j=$((j+1))
    done
    i=$((i+1))
done
