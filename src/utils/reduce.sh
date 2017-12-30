function remove_files {
folder=$1
number=$(($2 + 1))

find $folder -type f -print0 | sort -zR | tail -zn +$number | xargs -0 rm
}

if [ -v $1 ]
then
    echo "Please insert number of files to keep, exiting..." && exit 1
else
    number_of_files=$1
    echo "number of files in folder will be $number_of_files"
fi

directories=$(ls -d */)
for d in $directories
do
remove_files $d $number_of_files	
done




