"""
Script de Demostración de MiroFish-ES: Entregas por Dron
"""

import os
import sys

# Configurar codificación UTF-8 para consola Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import Config

def main():
    print("=" * 70)
    print("      DEMO MIROFISH-ES: SIMULACIÓN DE OPINIÓN PÚBLICA")
    print("=" * 70)
    
    # 1. Validar Configuración
    val = Config.validate()
    if val.get("errors"):
        print("[ERROR] Errores de configuración encontrados:", val["errors"])
        sys.exit(1)
    
    print("\n[✓] Configuración cargada correctamente desde .env")
    print(f"    - Modelo LLM: {Config.LLM_MODEL_NAME}")
    print(f"    - Base URL:   {Config.LLM_BASE_URL}")
    print(f"    - Memoria:    {Config.MEMORY_BACKEND}")
    
    # 2. Cargar Archivo Semilla
    seed_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo_seed.txt"))
    if not os.path.exists(seed_path):
        print(f"[ERROR] Archivo semilla no encontrado en: {seed_path}")
        sys.exit(1)
        
    with open(seed_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    print(f"\n[✓] Archivo Semilla Cargado: {os.path.basename(seed_path)} ({len(content)} caracteres)")
    print("-" * 50)
    print(content[:350] + "...\n[Contenido abreviado]")
    print("-" * 50)
    
    print("\n[🚀] La estructura de la demo está lista.")
    print("    Para ejecutar la simulación interactiva completa:")
    print("    1. Asegúrate de colocar tu LLM_API_KEY en el archivo .env")
    print("    2. Inicia el servidor backend y frontend ejecutando: npm run dev (o scripts\\iniciar.bat)")
    print("    3. Abre http://localhost:3000 y sube el archivo 'demo_seed.txt'")
    print("=" * 70)

if __name__ == "__main__":
    main()
