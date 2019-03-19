#!/bin/bash
git status
git add --all
echo -n 'Enter your commit message:'
read commitm
git commit -m "$commitm"
git push
