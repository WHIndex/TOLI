rm -f src/competitor/rmi/all_rmis.h

echo "#pragma once" >> src/competitor/rmi/all_rmis.h
for header in $(ls src/competitor/rmi/ | grep "\\.h$" | grep -v data | grep -v all_rmis ); do
    echo "#include \"${header}\"" >> src/competitor/rmi/all_rmis.h
done
