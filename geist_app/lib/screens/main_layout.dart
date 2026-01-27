import 'package:flutter/material.dart';
import '../utils/colors.dart';
import 'home_screen.dart';
import 'logs_screen.dart';
import 'settings_screen.dart'; // Ensure this file exists
import 'profile_screen.dart';

class MainLayout extends StatefulWidget {
  const MainLayout({super.key});

  @override
  State<MainLayout> createState() => _MainLayoutState();
}

class _MainLayoutState extends State<MainLayout> {
  int _currentIndex = 0;

  // Now we have 4 screens in the list
  final List<Widget> _screens = [
    const HomeScreen(), // Index 0
    const LogsScreen(), // Index 1
    const SettingsScreen(), // Index 2 (New Config Tab)
    const ProfileScreen(), // Index 3
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // IndexedStack keeps Home connection alive while you browse Settings
      body: IndexedStack(index: _currentIndex, children: _screens),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          boxShadow: [
            BoxShadow(color: Colors.black.withOpacity(0.1), blurRadius: 20),
          ],
        ),
        child: BottomNavigationBar(
          backgroundColor: Colors.white,
          currentIndex: _currentIndex,
          // Fixed type allows more than 3 items without shifting
          type: BottomNavigationBarType.fixed,
          selectedItemColor: AppColors.primary,
          unselectedItemColor: Colors.grey,
          showSelectedLabels: true,
          showUnselectedLabels: true,
          selectedFontSize: 12,
          unselectedFontSize: 12,
          onTap: (index) => setState(() => _currentIndex = index),
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.grid_view_rounded), // Modern "Home" icon
              label: "Home",
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.history_rounded),
              label: "Logs",
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.tune_rounded), // Sliders icon for Config
              label: "Config",
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.person_rounded),
              label: "Profile",
            ),
          ],
        ),
      ),
    );
  }
}
