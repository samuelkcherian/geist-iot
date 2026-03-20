// lib/widgets/radar_pulse.dart
import 'package:flutter/material.dart';

class RadarPulse extends StatefulWidget {
  final String status;
  const RadarPulse({super.key, required this.status});

  @override
  State<RadarPulse> createState() => _RadarPulseState();
}

class _RadarPulseState extends State<RadarPulse>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      lowerBound: 0.5,
      duration: const Duration(seconds: 2),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    Color pulseColor;
    IconData iconData;

    switch (widget.status) {
      case 'FALL':
        pulseColor = Colors.red;
        iconData = Icons.warning_rounded;
        break;
      case 'WALK':
        pulseColor = Colors.yellow.shade600;
        iconData = Icons.directions_walk;
        break;
      case 'SIT':
        pulseColor = Colors.blue;
        iconData = Icons.event_seat;
        break;
      case 'EMPTY':
      case 'Safe':
      default:
        pulseColor = Colors.greenAccent;
        iconData = Icons.radar;
        break;
    }

    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Stack(
          alignment: Alignment.center,
          children: [
            // Outer Ripple
            Container(
              width: 60 * _controller.value,
              height: 60 * _controller.value,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: pulseColor.withOpacity(1 - _controller.value),
              ),
            ),
            // Inner Circle
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: pulseColor,
                boxShadow: [
                  BoxShadow(color: pulseColor.withOpacity(0.6), blurRadius: 10),
                ],
              ),
              child: Icon(
                iconData,
                color: Colors.white,
                size: 24,
                shadows: const [
                  Shadow(
                    color: Colors.black45,
                    blurRadius: 4,
                    offset: Offset(0, 2),
                  ),
                ],
              ),
            ),
          ],
        );
      },
    );
  }
}
