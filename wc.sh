count1=$(pdftotext output.pdf - | tr -d '.' | wc -w)

total=$(expr $count1)

echo The current wordcount is $total... nerd
