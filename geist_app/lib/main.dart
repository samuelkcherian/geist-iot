// lib/main.dart
import 'package:flutter/material.dart';
import 'package:geist_app/screens/login_screen.dart';
import 'package:google_fonts/google_fonts.dart';
import 'utils/colors.dart';
// We will create this next!

void main() {
  runApp(const GeistApp());
}

class GeistApp extends StatelessWidget {
  const GeistApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Geist Monitor',
      theme: ThemeData(
        scaffoldBackgroundColor: AppColors.background,
        primaryColor: AppColors.primary,
        useMaterial3: true,
        // Apply the "Rubik" font globally to match the UI style
        textTheme: GoogleFonts.rubikTextTheme(Theme.of(context).textTheme),
      ),
      home: const LoginScreen(),
    );
  }
}
