# MiniRecorder Morphing Window Requirements

## 🎯 Core Requirements

### **Visual Behavior**
1. **Visualizer Always Centered**: The audio visualizer must remain in the exact same position throughout all animations
2. **Fixed Window Position**: The MiniRecorderView window position should never change during morphing
3. **Horizontal-Only Expansion**: Only width should change, height remains constant at 34px
4. **Hover-Triggered**: Expansion should occur on hover, collapse on hover exit

### **Layout States**

#### **Compact State (Default)**
- **Width**: ~70px (just enough for visualizer + minimal padding)
- **Content**: Audio visualizer/status display only
- **Buttons**: Hidden/not rendered
- **Centering**: Visualizer perfectly centered in compact window

#### **Expanded State (On Hover)**
- **Width**: ~160px (current full width)
- **Content**: RecorderPromptButton + Visualizer + RecorderPowerModeButton
- **Buttons**: Fully visible and functional
- **Centering**: Visualizer remains in same absolute screen position

## 🔧 Technical Constraints

### **Window Positioning**
- Window's center point must remain constant
- When expanding from 70px → 160px, window should grow equally left and right (45px each side)
- `NSRect` calculations must account for center-anchored growth

### **Animation Requirements**
- Smooth spring animation (~0.3-0.4s duration)
- Buttons should appear/disappear gracefully (fade in/out or slide from edges)
- No jarring movements or position jumps
- Reversible animation (expand ↔ collapse)

### **SwiftUI Layout Considerations**
- HStack with conditional button rendering
- Visualizer maintains `frame(maxWidth: .infinity)` behavior in both states
- Proper spacing and padding calculations for both states

## 💡 Recommended Implementation Strategy

### **Approach: Center-Anchored Window Growth with Sliding Buttons**

#### **Window Management (MiniRecorderPanel)**
```
Compact Window Rect:
- Width: 70px
- Height: 34px  
- X: screenCenter - 35px
- Y: current Y position

Expanded Window Rect:
- Width: 160px
- Height: 34px
- X: screenCenter - 80px  // Grows left by 45px
- Y: same Y position      // Grows right by 45px
```

#### **SwiftUI Layout (MiniRecorderView)**
```
HStack(spacing: 0) {
    // Left button - slides in from left edge
    if isExpanded {
        RecorderPromptButton()
            .transition(.move(edge: .leading).combined(with: .opacity))
    }
    
    // Visualizer - always centered, never moves
    statusView
        .frame(width: visualizerWidth) // Fixed width
    
    // Right button - slides in from right edge  
    if isExpanded {
        RecorderPowerModeButton()
            .transition(.move(edge: .trailing).combined(with: .opacity))
    }
}
```

#### **State Management**
- `@State private var isExpanded = false`
- `@State private var isHovering = false`
- Hover detection with debouncing for smooth UX
- Window resize triggered by state changes

## 🎨 Animation Sequence

### **Expansion (Compact → Expanded)**
1. **Trigger**: Mouse enters window bounds
2. **Window**: Animate width 70px → 160px (center-anchored)
3. **Buttons**: Slide in from edges with fade-in
4. **Duration**: ~0.3s with spring easing
5. **Result**: Visualizer appears unmoved, buttons visible

### **Collapse (Expanded → Compact)**
1. **Trigger**: Mouse leaves window bounds (with delay)
2. **Buttons**: Slide out to edges with fade-out
3. **Window**: Animate width 160px → 70px (center-anchored)
4. **Duration**: ~0.3s with spring easing
5. **Result**: Back to visualizer-only, same position

## 🚫 Critical Don'ts

- **Never move the visualizer's absolute screen position**
- **Never change the window's center point**
- **Never animate height or vertical position**
- **Never show jarring button pop-ins (use smooth transitions)**
- **Never let buttons overlap the visualizer during animation**

## 📐 Calculations

### **Visualizer Dimensions**
- AudioVisualizer: 12 bars × 3px + 11 × 2px spacing = 58px width
- With padding: ~70px total compact width

### **Button Dimensions**  
- Each button: ~24px width + padding
- Total button space: ~90px (45px per side)
- Total expanded width: 70px + 90px = 160px

### **Center-Anchored Growth**
```
Compact X position: screenCenterX - 35px
Expanded X position: screenCenterX - 80px
Growth: 45px left + 45px right = 90px total
```

## 🎯 Success Criteria

✅ **Visualizer never visually moves during any animation**  
✅ **Window position anchor point remains constant**  
✅ **Smooth hover-based expansion/collapse**  
✅ **Buttons appear/disappear gracefully**  
✅ **No layout jumps or glitches**  
✅ **Maintains current functionality in expanded state**

This approach ensures the visualizer appears completely stationary while the window "grows around it" to reveal the buttons, creating a seamless morphing effect.