import React, { useState, useEffect, useRef, useCallback } from 'react';
import { StyleSheet, Text, View, ScrollView, TouchableOpacity, Switch } from 'react-native';
import './index.css'; // wtf fff

export default function App() {
  // Theme state
  const [isDarkMode, setIsDarkMode] = useState<boolean>(false);

  // State for German transcription
  const [currentLineGerman, setCurrentLineGerman] = useState<string>("");
  const [previousLinesGerman, setPreviousLinesGerman] = useState<string[]>([]);

  // State for English translation
  const [currentLineEnglish, setCurrentLineEnglish] = useState<string>("");
  const [previousLinesEnglish, setPreviousLinesEnglish] = useState<string[]>([]);

  // Refs for ScrollViews
  const scrollViewGermanRef = useRef<ScrollView>(null);
  const scrollViewEnglishRef = useRef<ScrollView>(null);

  // State to track auto-scroll behavior
  const [isAutoScrollGerman, setIsAutoScrollGerman] = useState<boolean>(true);
  const [isAutoScrollEnglish, setIsAutoScrollEnglish] = useState<boolean>(true);

  // New message indicators
  const [newMessagesGerman, setNewMessagesGerman] = useState<boolean>(false);
  const [newMessagesEnglish, setNewMessagesEnglish] = useState<boolean>(false);

  // Timers for auto-scroll re-enabling
  const autoScrollTimerGerman = useRef<NodeJS.Timeout | null>(null);
  const autoScrollTimerEnglish = useRef<NodeJS.Timeout | null>(null);

  // Ref to store the WebSocket instance
  const wsRef = useRef<WebSocket | null>(null);

  // Function to handle theme toggle
  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  // Function to export German Transcription
  const exportGermanTranscription = () => {
    const element = document.createElement("a");
    const file = new Blob([previousLinesGerman.join('\n')], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = "German_Transcription.txt";
    document.body.appendChild(element); // Required for this to work in FireFox
    element.click();
    document.body.removeChild(element);
  };

  // Function to export English Translation
  const exportEnglishTranslation = () => {
    const element = document.createElement("a");
    const file = new Blob([previousLinesEnglish.join('\n')], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = "English_Translation.txt";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  // Initialize WebSocket once
  useEffect(() => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';

    const wsPort = '7000'; 
    const ws = new WebSocket(`${wsProtocol}://${window.location.hostname}:${wsPort}/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const germanText = data.transcription;
        const englishText = data.translation;

        setCurrentLineGerman(germanText);
        setPreviousLinesGerman(prevLines => [...prevLines, germanText]);

        setCurrentLineEnglish(englishText);
        setPreviousLinesEnglish(prevLines => [...prevLines, englishText]);

        
        if (!isAutoScrollGerman) {
          setNewMessagesGerman(true);
        }
        if (!isAutoScrollEnglish) {
          setNewMessagesEnglish(true);
        }
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      ws.close();
    };

    return () => {
      ws.close();
    };
  }, []); 

  // Auto-scroll to the latest message in German transcription
  useEffect(() => {
    if (isAutoScrollGerman && scrollViewGermanRef.current) {
      scrollViewGermanRef.current.scrollToEnd({ animated: true });
      setNewMessagesGerman(false);
    }
  }, [previousLinesGerman, isAutoScrollGerman]);

  // Auto-scroll to the latest message in English translation
  useEffect(() => {
    if (isAutoScrollEnglish && scrollViewEnglishRef.current) {
      scrollViewEnglishRef.current.scrollToEnd({ animated: true });
      setNewMessagesEnglish(false);
    }
  }, [previousLinesEnglish, isAutoScrollEnglish]);

  // Handler for German ScrollView scroll events
  const handleScrollGerman = useCallback((event: any) => {
    const { layoutMeasurement, contentOffset, contentSize } = event.nativeEvent;
    const isAtBottom = layoutMeasurement.height + contentOffset.y >= contentSize.height - 20;

    if (isAtBottom) {
      setIsAutoScrollGerman(true);
      setNewMessagesGerman(false);
    } else {
      setIsAutoScrollGerman(false);
      setNewMessagesGerman(false);
    }

    if (isAutoScrollGerman) {
      if (autoScrollTimerGerman.current) {
        clearTimeout(autoScrollTimerGerman.current);
      }

      autoScrollTimerGerman.current = setTimeout(() => {
        setIsAutoScrollGerman(true);
      }, 3000); // 3 seconds of inactivity
    }
  }, [isAutoScrollGerman]);

  // Handler for English ScrollView scroll events
  const handleScrollEnglish = useCallback((event: any) => {
    const { layoutMeasurement, contentOffset, contentSize } = event.nativeEvent;
    const isAtBottom = layoutMeasurement.height + contentOffset.y >= contentSize.height - 20;

    if (isAtBottom) {
      setIsAutoScrollEnglish(true);
      setNewMessagesEnglish(false);
    } else {
      setIsAutoScrollEnglish(false);
      setNewMessagesEnglish(false);
    }

    if (isAutoScrollEnglish) {
      if (autoScrollTimerEnglish.current) {
        clearTimeout(autoScrollTimerEnglish.current);
      }

      autoScrollTimerEnglish.current = setTimeout(() => {
        setIsAutoScrollEnglish(true);
      }, 3000); // 3 seconds of inactivity
    }
  }, [isAutoScrollEnglish]);

  return (
    <View style={[styles.container, isDarkMode ? styles.darkContainer : styles.lightContainer]}>
      {/* Header */}
      <View style={[styles.header, isDarkMode ? styles.darkHeader : styles.lightHeader]}>
        <Text style={[styles.headerTitle, isDarkMode ? styles.darkText : styles.lightText]}>Transcribe & Translate</Text>
        <View style={styles.themeToggle}>
          <Text style={[styles.themeText, isDarkMode ? styles.darkText : styles.lightText]}>
            {isDarkMode ? "Dark Mode" : "Light Mode"}
          </Text>
          <Switch
            value={isDarkMode}
            onValueChange={toggleTheme}
            thumbColor={isDarkMode ? "#f4f3f4" : "#f4f3f4"}
            trackColor={{ false: "#767577", true: "#81b0ff" }}
          />
        </View>
      </View>

      {/* Main Content */}
      <View style={styles.mainContent}>
        {/* German Transcription Section */}
        <View style={[styles.section, isDarkMode ? styles.darkSection : styles.lightSection]}>
          <View style={styles.sectionHeader}>
            <Text style={[styles.sectionTitle, isDarkMode ? styles.darkText : styles.lightText]}>German Transcription</Text>
            <TouchableOpacity onPress={exportGermanTranscription} style={styles.exportButton}>
              <Text style={styles.exportButtonText}>Export</Text>
            </TouchableOpacity>
          </View>
          <ScrollView
            ref={scrollViewGermanRef}
            contentContainerStyle={styles.historyContainer}
            style={[styles.scrollView, isDarkMode ? styles.darkScrollView : styles.lightScrollView]}
            showsVerticalScrollIndicator={false}
            onScroll={handleScrollGerman}
            scrollEventThrottle={16}
            className="scrollView" 
          >
            {previousLinesGerman.map((line, index) => (
              <Text key={index} style={[styles.previousLine, isDarkMode ? styles.darkText : styles.lightText]}>
                {line}
              </Text>
            ))}
          </ScrollView>
          <View style={[styles.currentContainer, isDarkMode ? styles.darkCurrentContainer : styles.lightCurrentContainer]}>
            <Text style={[styles.currentLine, isDarkMode ? styles.darkText : styles.lightText]}>
              {currentLineGerman}
            </Text>
          </View>
          {newMessagesGerman && (
            <TouchableOpacity
              style={styles.newMessageButton}
              onPress={() => {
                if (scrollViewGermanRef.current) {
                  scrollViewGermanRef.current.scrollToEnd({ animated: true });
                }
              }}
            >
              <Text style={styles.newMessageText}>New Messages</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* English Translation Section */}
        <View style={[styles.section, isDarkMode ? styles.darkSection : styles.lightSection]}>
          <View style={styles.sectionHeader}>
            <Text style={[styles.sectionTitle, isDarkMode ? styles.darkText : styles.lightText]}>English Translation</Text>
            <TouchableOpacity onPress={exportEnglishTranslation} style={styles.exportButton}>
              <Text style={styles.exportButtonText}>Export</Text>
            </TouchableOpacity>
          </View>
          <ScrollView
            ref={scrollViewEnglishRef}
            contentContainerStyle={styles.historyContainer}
            style={[styles.scrollView, isDarkMode ? styles.darkScrollView : styles.lightScrollView]}
            showsVerticalScrollIndicator={false}
            onScroll={handleScrollEnglish}
            scrollEventThrottle={16}
            className="scrollView"
          >
            {previousLinesEnglish.map((line, index) => (
              <Text key={index} style={[styles.previousLine, isDarkMode ? styles.darkText : styles.lightText]}>
                {line}
              </Text>
            ))}
          </ScrollView>
          <View style={[styles.currentContainer, isDarkMode ? styles.darkCurrentContainer : styles.lightCurrentContainer]}>
            <Text style={[styles.currentLine, isDarkMode ? styles.darkText : styles.lightText]}>
              {currentLineEnglish}
            </Text>
          </View>
          {newMessagesEnglish && (
            <TouchableOpacity
              style={styles.newMessageButton}
              onPress={() => {
                if (scrollViewEnglishRef.current) {
                  scrollViewEnglishRef.current.scrollToEnd({ animated: true });
                }
              }}
            >
              <Text style={styles.newMessageText}>New Messages</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    fontFamily: 'Roboto, sans-serif',
  },
  lightContainer: {
    backgroundColor: '#f0f4f7',
  },
  darkContainer: {
    backgroundColor: '#121212',
  },
  header: {
    padding: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottomWidth: 1,
  },
  lightHeader: {
    backgroundColor: '#ffffff',
    borderBottomColor: '#dddddd',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  darkHeader: {
    backgroundColor: '#1e1e1e',
    borderBottomColor: '#333333',
    shadowColor: '#fff',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 3,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '700',
  },
  themeToggle: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  themeText: {
    marginRight: 8,
    fontSize: 16,
  },
  lightText: {
    color: '#333333',
  },
  darkText: {
    color: '#ffffff',
  },
  mainContent: {
    flex: 1,
    flexDirection: 'row', // Split the screen horizontally
    padding: 10,
    overflow: 'hidden',
  },
  section: {
    flex: 1,
    margin: 10,
    padding: 20,
    borderRadius: 10,
    borderWidth: 1,
    position: 'relative',
  },
  lightSection: {
    backgroundColor: '#ffffff',
    borderColor: '#dddddd',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  darkSection: {
    backgroundColor: '#1e1e1e',
    borderColor: '#333333',
    shadowColor: '#fff',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
  },
  exportButton: {
    backgroundColor: '#4a90e2',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 5,
  },
  exportButtonText: {
    color: '#ffffff',
    fontSize: 14,
  },
  historyContainer: {
    flexGrow: 1,
    justifyContent: 'flex-end',
    paddingBottom: 10,
  },
  previousLine: {
    fontSize: 16,
    marginBottom: 8,
    lineHeight: 24,
  },
  currentContainer: {
    padding: 15,
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: '#dddddd',
    borderRadius: 8,
    marginTop: 10,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.02,
    shadowRadius: 2,
    elevation: 1,
  },
  lightCurrentContainer: {
    backgroundColor: '#f9f9f9',
  },
  darkCurrentContainer: {
    backgroundColor: '#2c2c2c',
    borderTopColor: '#444444',
  },
  currentLine: {
    fontSize: 18,
    fontWeight: '500',
  },
  lightScrollView: {
    backgroundColor: '#f0f4f7',
  },
  darkScrollView: {
    backgroundColor: '#2c2c2c',
  },
  scrollView: {
    borderRadius: 8,
    padding: 10,
    maxHeight: '60vh',
    overflow: 'hidden',
  },
  newMessageButton: {
    position: 'absolute',
    bottom: 20,
    left: '50%',
    transform: [{ translateX: -75 }],
    backgroundColor: '#4a90e2',
    paddingVertical: 8,
    paddingHorizontal: 15,
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.15,
    shadowRadius: 4,
    elevation: 5,
  },
  newMessageText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '500',
  },
});
