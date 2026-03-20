import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import '../utils/colors.dart';
import '../services/socket_service.dart';
import '../services/notification_service.dart';
import '../widgets/radar_pulse.dart'; // Import the new widget

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String _statusText = "Initializing...";
  String _currentStatus = "Safe";
  bool _isConnected = false;

  late SocketService _socketService;
  final NotificationService _notificationService = NotificationService();

  @override
  void initState() {
    super.initState();
    _notificationService.init();
    _socketService = SocketService(
      onStatusUpdate: _updateUI,
      onConnect: _handleConnectionSuccess,
    );
    _socketService.initConnection();
  }

  Future<void> _makeEmergencyCall() async {
    final Uri launchUri = Uri(scheme: 'tel', path: '108');
    if (await canLaunchUrl(launchUri)) {
      await launchUrl(launchUri);
    }
  }

  void _handleConnectionSuccess() {
    if (mounted) {
      setState(() {
        _isConnected = true;
        if (_currentStatus != "FALL") {
          _statusText = "Active Monitoring";
        }
      });
    }
  }

  void _updateUI(Map<String, dynamic> data) {
    if (!mounted) return;
    setState(() {
      String status = data['status'];
      _currentStatus = status;
      if (status == "FALL") {
        _statusText = "CRITICAL FALL DETECTED";
        _notificationService.showFallAlert();
      } else if (status == "WALK") {
        _statusText = "Subject Walking";
      } else if (status == "SIT") {
        _statusText = "Subject Seated";
      } else {
        _statusText = "Room Empty";
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      floatingActionButton: _currentStatus == "FALL"
          ? FloatingActionButton.extended(
              onPressed: _makeEmergencyCall,
              backgroundColor: AppColors.alertRed,
              elevation: 10,
              icon: const Icon(Icons.phone_in_talk, color: Colors.white),
              label: const Text(
                "CALL 108",
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.2,
                ),
              ),
            )
          : null,

      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // --- HEADER ---
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // CLEAN "GEIST" BRANDING
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10,
                          vertical: 5,
                        ),
                        decoration: BoxDecoration(
                          color: AppColors.primary,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: const Text(
                          "GEIST",
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.w900,
                            letterSpacing: 3.0, // Spaced out for high-tech look
                            fontSize: 14,
                          ),
                        ),
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        "Safety Dashboard",
                        style: TextStyle(
                          fontSize: 26,
                          fontWeight: FontWeight.w800,
                          color: AppColors.darkText,
                          letterSpacing: -0.5,
                        ),
                      ),
                    ],
                  ),

                  // LOGO AVATAR
                  Container(
                    padding: const EdgeInsets.all(3),
                    decoration: const BoxDecoration(
                      color: Colors.white,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(color: Colors.black12, blurRadius: 10),
                      ],
                    ),
                    child: const CircleAvatar(
                      radius: 24,
                      backgroundColor: Colors.white,
                      // SWITCHED TO ASSET IMAGE
                      backgroundImage: AssetImage("assets/logo.png"),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 30),

              // --- MAIN STATUS CARD (Gradient) ---
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(25),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: _getGradientColors(),
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(30),
                  boxShadow: [
                    BoxShadow(
                      color: _getShadowColor(),
                      blurRadius: 20,
                      offset: const Offset(0, 10),
                    ),
                  ],
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 10,
                              vertical: 5,
                            ),
                            decoration: BoxDecoration(
                              color: Colors.white.withOpacity(0.2),
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: const Text(
                              "LIVE STATUS",
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 10,
                                fontWeight: FontWeight.bold,
                                shadows: [
                                  Shadow(color: Colors.black38, blurRadius: 2, offset: Offset(0, 1)),
                                ],
                              ),
                            ),
                          ),
                          const SizedBox(height: 15),
                          Text(
                            _statusText,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 22,
                              height: 1.2,
                              fontWeight: FontWeight.bold,
                              shadows: [
                                Shadow(color: Colors.black45, blurRadius: 4, offset: Offset(0, 2)),
                              ],
                            ),
                          ),
                          const SizedBox(height: 5),
                          Text(
                            _isConnected ? "System Online" : "Connecting...",
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.9),
                              fontSize: 12,
                              fontWeight: FontWeight.w500,
                              shadows: const [
                                Shadow(color: Colors.black38, blurRadius: 2, offset: Offset(0, 1)),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                    // PULSE ANIMATION
                    Container(
                      height: 80,
                      width: 80,
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.1),
                        shape: BoxShape.circle,
                      ),
                      child: Center(
                        child: _isConnected
                            ? RadarPulse(status: _currentStatus)
                            : const CircularProgressIndicator(
                                color: Colors.white,
                              ),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 35),

              // --- METRICS GRID ---
              const Text(
                "Real-time Metrics",
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: AppColors.darkText,
                ),
              ),
              const SizedBox(height: 20),

              Row(
                children: [
                  Expanded(
                    child: _buildMetricCard(
                      "Sensors",
                      "Active",
                      Icons.sensors,
                      Colors.blueAccent,
                    ),
                  ),
                  const SizedBox(width: 15),
                  Expanded(
                    child: _buildMetricCard(
                      "Battery",
                      "92%",
                      Icons.battery_charging_full,
                      Colors.green,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 15),
              _buildWideCard(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildMetricCard(
    String title,
    String value,
    IconData icon,
    Color color,
  ) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.05),
            blurRadius: 15,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 28),
          const SizedBox(height: 15),
          Text(title, style: TextStyle(color: Colors.grey[500], fontSize: 12)),
          const SizedBox(height: 5),
          Text(
            value,
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 18,
              color: AppColors.darkText,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildWideCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.05),
            blurRadius: 15,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.purple.withOpacity(0.1),
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Icon(Icons.wifi_tethering, color: Colors.purple),
          ),
          const SizedBox(width: 15),
          const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                "Connectivity",
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
              ),
              Text(
                "Strong Signal (5ms)",
                style: TextStyle(color: Colors.grey, fontSize: 12),
              ),
            ],
          ),
        ],
      ),
    );
  }

  List<Color> _getGradientColors() {
    if (_currentStatus == "FALL") {
      return [AppColors.alertRed, AppColors.alertOrange];
    } else if (_currentStatus == "WALK") {
      return [Colors.amber.shade700, Colors.amber.shade400];
    } else if (_currentStatus == "SIT") {
      return [Colors.blue.shade700, Colors.blue.shade400];
    } else {
      // Empty: Dark slate gray to differentiate from the blue 'Sit' state
      return [const Color(0xFF2C3E50), const Color(0xFF4CA1AF)];
    }
  }

  Color _getShadowColor() {
    if (_currentStatus == "FALL") {
      return AppColors.alertRed.withOpacity(0.4);
    } else if (_currentStatus == "WALK") {
      return Colors.amber.shade600.withOpacity(0.4);
    } else if (_currentStatus == "SIT") {
      return Colors.blue.withOpacity(0.4);
    } else {
      return const Color(0xFF2C3E50).withOpacity(0.4);
    }
  }
}
