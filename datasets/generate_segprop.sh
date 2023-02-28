big=("0043" "0044" "0045" "0046" "0047" "0050" "0053" "0085" "0093" "0097" "0101" "0114" "0118" )
times=(4      1      1      2      1      2      1      1      1      5      1      2      3)

CHUNK=50
i=0
prefix=tfm
for vid in ${big[*]}
do
    iters=${times[i]}
    echo $iters
    ini=0
    end=$CHUNK
    j=0
    while ((j < iters))
    do
        python3 ~/$prefix/drone_landing/datasets/generate_flow_db.py -vid="$vid" -start="$ini" -end="$end"
        python3 ~/$prefix/segprop2/ruralscapes_demo.py -vid="$vid" -start_from="$ini" -stop_at="$end"
        python3 ~/$prefix/drone_landing/datasets/sparse_to_segmentation.py
        rm ~/$prefix/ruralscapes/flow_farneback/*
        rm ~/$prefix/ruralscapes/output_2k/*
        ini=$((ini+CHUNK))
        end=$((end+CHUNK))
        j=$((j+1))
    done
    i=$((i+1))
done
