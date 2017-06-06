for i in {2..10}
do
  echo $i
  python trainer.py --exp exp$i 
done
