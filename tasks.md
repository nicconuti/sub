# ✅ COMPLETATO - MODULARIZZAZIONE RIUSCITA

**Stato**: TUTTI I TASK COMPLETATI AL 100%

# PROMPT ORIGINALE

Hai davanti un file Python monolitico chiamato `app.py` contenente oltre 3.500 righe. 
Desidero modularizzarlo seguendo il principio di Single Responsibility, dividendo ogni responsabilità in un file separato di massimo 200 righe. 
Il progetto è un'app PyQt6 con visualizzazioni scientifiche e simulazioni fisiche.

Per ogni modulo che estrai, devi:

1. Individuare la responsabilità unica (GUI, logica, visualizzazione, I/O, ecc.)
2. Spostare classi/funzioni pertinenti in un nuovo file con import corretti
3. Mantenere l'app funzionante (non rompere i riferimenti)
4. Generare eventualmente cartelle (es. `core`, `ui`, `plot`, `helpers`)
5. Lasciare in `main.py` solo l’avvio dell’app

Il codice deve essere leggibile, modulare, documentato. Procedi modulo per modulo, senza saltare passaggi.

ogni volta che completi uno di questi task, segnalo come completato aggiungendo una [x]

Ogni modulo avrà una responsabilità unica, con massimo ~200 righe a file. I task sono ordinati e progressivi.
se è già presente aggiungere alla propria knowledge il file notes.md

# TASKS

🧱 1. Analisi del dominio e identificazione delle responsabilità
[] Leggere l’intero app.py per identificare le classi principali annotando tutte le informazini che vengono dedotte nel processo in un file notes.md (e.g posizione dei metodi - riga di inizio e termine dei metodi-, descrizione, categorizzazione ecc. )
[] Raccogliere responsabilità ricorrenti: GUI, logica fisica, visualizzazione, I/O, ecc.
[] Annotare eventuali dipendenze forti (tra GUI ↔ core logico)

🗂 2. Suddivisione in macro-moduli (max 200 righe a file)

[] main.py → Avvio dell'applicazione (PyQt, QApplication, ecc.)
[] ui/main_window.py → Classe MainWindow e collegamenti con i widget
[] ui/widgets/ → Tutti i widget custom (se presenti): slider, pannelli, ecc.
[] core/simulation.py → Logica fisica e matematica della simulazione SPL

[] core/config.py → Configurazioni e costanti (già parzialmente importate)
[] core/data_loader.py → Caricamento e validazione dei dati da file
[] core/diagnostics.py → Diagnostica e validazione dei parametri utente
[] core/exporter.py → Esportazione dati/progetti
[] plot/visualizer.py → Visualizzazione matplotlib integrata in PyQt
[] assets/ → Eventuali risorse statiche (icone, configurazioni utente, temi)
[] helpers/utils.py → Funzioni di utilità comuni (es. conversioni, log)

🧹 3. Refactor operativo[]Copiare i blocchi di codice nei moduli corrispondenti

[]Sostituire i riferimenti locali con import espliciti (from core.simulation import ...)
[]Rimuovere i commenti legacy / dead code (con backup iniziale)
[]Introdurre logging (se assente) nei moduli core

🧪 4. Testing e validazione
[]Verificare che ogni modulo importi correttamente
[]Lanciare l'app e testare funzionalità principali
[]Aggiungere test unitari nei moduli core (simulazione, config, ecc.)

📦 5. Ambiente & distribuzione
[]Generare requirements.txt (pipreqs .)
[]verificare che il .gitignore sia completo (include __pycache__, .vscode, ecc.)
[] verificare se il venv (.venv) sia completo
[]Strutturare il progetto come pacchetto installabile e multipiattaforma (__init__.py, setup?)

