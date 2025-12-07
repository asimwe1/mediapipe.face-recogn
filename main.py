#!/usr/bin/env python3
"""
Face Recognition Pipeline - Main Entry Point
Orchestrates capture, training, and prediction workflows
"""

import sys
import os

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')

def display_menu():
    """Display main menu"""
    clear_screen()
    print("=" * 50)
    print("  Face Recognition Pipeline")
    print("  MediaPipe + LBPH")
    print("=" * 50)
    print("\n1. Capture Face Images")
    print("2. Train Model")
    print("3. Run Face Recognition")
    print("4. Exit")
    print("\n" + "=" * 50)

def capture_faces():
    """Run face capture module"""
    from src.capture import run_capture
    run_capture()

def train_model():
    """Run training module"""
    from src.train import run_training
    run_training()

def predict_faces():
    """Run prediction module"""
    from src.predict import run_prediction
    run_prediction()

def main():
    """Main application loop"""
    while True:
        display_menu()
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            capture_faces()
            input("\nPress Enter to continue...")
        elif choice == '2':
            train_model()
            input("\nPress Enter to continue...")
        elif choice == '3':
            predict_faces()
            input("\nPress Enter to continue...")
        elif choice == '4':
            print("\nExiting... Goodbye!")
            sys.exit(0)
        else:
            print("\nInvalid option. Please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
