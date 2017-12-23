find ./8PSK -type f -print0 | sort -zR | tail -zn +201 | xargs -0 rm
