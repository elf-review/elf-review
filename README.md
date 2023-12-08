## ELVES Compression Reproducing
```
pip3 install -r requirements.txt
./config
make
```
```
python3 main.py
```

## Evaluation and Comparison Reproducing
Requires Java 9 or higher
```
sudo apt install maven
```
```
python3 evaluation.py
```

## ELF Performance

```
./elf_pthread -i original_weights_path.bin -p weight_type -n para_num -c (-c for compression, -d for decompression)
```

## ELVES Fuzz Testing
The fuzz testing programs and results for the 300 models are located in the folder: elf-review/ELF/fuzz_testing/

<!--## Evaluation Reproducing
python3 main.py
-->

<!--
**elf-review/elf-review** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
