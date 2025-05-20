import os
import subprocess
import toml

# Path al pyproject.toml
CONFIG_FILE = 'pyproject.toml'

# Funzione per aumentare la versione
def bump_version():
    # Leggi il file pyproject.toml
    data = toml.load(CONFIG_FILE)
    
    # Naviga nella struttura fino alla versione
    if 'tool' not in data or 'setuptools' not in data['tool'] or 'version' not in data['tool']['setuptools']:
        raise ValueError("❌ Errore: la versione non è presente nel file pyproject.toml")
    
    version = data['tool']['setuptools']['version']
    major, minor, patch = map(int, version.split('.'))
    
    # Incrementa la versione di patch (0.1.0 -> 0.1.1)
    patch += 1
    new_version = f"{major}.{minor}.{patch}"
    
    # Aggiorna la versione nel file
    data['tool']['setuptools']['version'] = new_version
    with open(CONFIG_FILE, 'w') as f:
        toml.dump(data, f)
    
    print(f"🔄 Versione aggiornata a: {new_version}")
    return new_version

# Funzione per pulire le vecchie build
def clean_build():
    print("🧹 Pulizia vecchie build...")
    if os.path.exists('dist'):
        os.system('rm -rf dist')
    if os.path.exists('build'):
        os.system('rm -rf build')
    for d in os.listdir():
        if d.endswith('.egg-info'):
            os.system(f'rm -rf {d}')

# Funzione per costruire il pacchetto
def build_package():
    print("🔨 Costruzione del pacchetto...")
    subprocess.run(['python3', '-m', 'build'])

# Funzione per pubblicare su PyPI
def upload_to_pypi():
    print("🚀 Pubblicazione su PyPI...")
    subprocess.run(['twine', 'upload', '--repository', 'pypi', 'dist/*'])

# Esecuzione
if __name__ == '__main__':
    print("✨ Inizio deploy...")
    new_version = bump_version()
    clean_build()
    build_package()
    upload_to_pypi()
    print(f"✅ Versione {new_version} pubblicata con successo!")
